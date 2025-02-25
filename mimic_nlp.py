#!/usr/bin/env python3
"""
This script implements an extended NLP pipeline for processing MIMIC-III clinical notes.
It loads data from a DuckDB database, extracts entities using spaCy, SciSpacy, and MedSpaCy,
trains custom Word2Vec embeddings and/or loads pre-trained medical embeddings,
analyzes entity relationships through dependency parsing, and provides multiple
visualization techniques including displaCy for entity and dependency visualization.

Analytical Focus:
    Sepsis & Infectious Diseases using ICD-9 codes:
      - 995.91 (Sepsis)
      - 038.x (Bacteremia & bloodstream infections)
      - 486 (Pneumonia)

Author: Tim Frenzel
License: MIT
Version: 1.14
Created: February-2025

Usage:
    python mimic_nlp.py --db_path "path/to/mimic.duckdb" 
                        --icd9_codes "995.91,038,486" 
                        --note_category "Nursing/other" 
                        --embedding_choice both 
                        --visualize_entities
                        --extract_relationships
                        --output_path results
                        --use_medspacy

Dependencies:
    - Python 3.8+
    - duckdb, pandas, numpy
    - spacy, scispacy, medspacy
    - gensim, matplotlib, scikit-learn, tqdm
    - networkx, seaborn (for visualization)
    - pycontext (for negation detection)
    - (Optional: umap-learn for advanced dimensionality reduction)
    
Inputs:
    - DuckDB database file containing MIMIC data.

Outputs:
    - Processed notes, extracted entities, entity visualizations, 
      relationship knowledge graphs, and embedding visualizations.
    
License:
    MIT License - See LICENSE file for details.
"""

# Global parameters for project scale
GLOBAL_MAX_NOTES = 5000          # Default maximum notes to process
GLOBAL_BATCH_SIZE = 32          # Default batch size for document processing
GLOBAL_NOTE_CATEGORY = "Nursing/other"  # Default note category

import os
import sys
import duckdb
import re
import spacy
import torch
import gensim
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import gc
import warnings
import random
from pathlib import Path
from typing import List, Tuple, Dict, Union, Optional, Any
from tqdm.auto import tqdm
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as gensim_downloader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime
from packaging import version
import pydantic

# Optional imports with fallbacks
try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    warnings.warn("seaborn not installed; some visualizations will be limited.")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    warnings.warn("networkx not installed; graph visualizations will be disabled.")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    warnings.warn("umap-learn not installed; PCA and t-SNE will be used for dimensionality reduction.")

try:
    import pycontext
    from pycontext.pycontext import PyConText
    HAS_PYCONTEXT = True
except ImportError:
    HAS_PYCONTEXT = False
    warnings.warn("pycontext not installed; negation detection will be disabled.")

# New import for MedSpaCy
try:
    import medspacy
    HAS_MEDSPACY = True
except ImportError:
    HAS_MEDSPACY = False
    warnings.warn("medspacy not installed; medspacy features will be disabled. Run: pip install medspacy")

# --- Pydantic version check ---
pydantic_version = getattr(pydantic, "__version__", "1.11")
if version.parse(pydantic_version) >= version.parse("2.0"):
    sys.exit("Incompatible pydantic version detected. Please use a version less than 2.0.")

# ------------------- Pipeline Class ------------------- #
class MIMICNLPPipeline:
    """
    A comprehensive pipeline for processing MIMIC-III clinical notes.
    """
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.setup_logging()
        try:
            self.initialize_models()
        except Exception as e:
            sys.exit(f"Failed to initialize NLP models: {str(e)}")

    def setup_logging(self) -> None:
        try:
            logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                level=logging.INFO if self.args.verbose else logging.WARNING
            )
            self.logger = logging.getLogger(__name__)
        except Exception as e:
            sys.exit(f"Failed to setup logging: {str(e)}")

    def initialize_models(self) -> None:
        """Initialize NLP models, retaining parser for dependency analysis."""
        self.logger.info("Initializing NLP models...")
        current_spacy_version = spacy.__version__
        self.logger.info(f"Using spaCy version {current_spacy_version}")

        if torch.cuda.is_available() and not self.args.disable_gpu:
            try:
                memory_fraction = max(0.1, min(1.0, self.args.gpu_memory_fraction))
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                self.logger.info(f"Set PyTorch to use {memory_fraction*100:.0f}% of GPU memory")
                spacy.prefer_gpu()
                self.logger.info(f"GPU acceleration enabled for spaCy {current_spacy_version}")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            except Exception as e:
                self.logger.warning(f"GPU initialization failed: {str(e)}")
                self.logger.warning("Falling back to CPU.")
        else:
            if self.args.disable_gpu:
                self.logger.info("GPU acceleration disabled by user request.")
            else:
                self.logger.info("No GPU detected, running on CPU.")

        try:
            self.logger.info("Loading en_core_web_sm model...")
            self.spacy_model = spacy.load("en_core_web_sm")
            # Keep parser if relationship extraction is enabled
            if not self.args.extract_relationships and "parser" in self.spacy_model.pipe_names:
                self.spacy_model.remove_pipe("parser")
            self.logger.info("SpaCy model loaded successfully")
        except OSError as e:
            sys.exit(f"Failed to load 'en_core_web_sm': {str(e)}")

        try:
            self.logger.info("Loading en_ner_bc5cdr_md model...")
            self.scispacy_model = spacy.load("en_ner_bc5cdr_md")
            # Keep parser if relationship extraction is enabled
            if not self.args.extract_relationships and "parser" in self.scispacy_model.pipe_names:
                self.scispacy_model.remove_pipe("parser")
            self.logger.info("SciSpacy model loaded successfully")
        except OSError as e:
            sys.exit(f"Failed to load 'en_ner_bc5cdr_md': {str(e)}")

        # Initialize MedSpaCy if requested and available
        self.medspacy_model = None
        if self.args.use_medspacy and HAS_MEDSPACY:
            try:
                self.logger.info("Initializing MedSpaCy model...")
                # Create MedSpaCy pipeline with clinical components
                self.medspacy_model = medspacy.load(enable=["medspacy_tokenizer", "medspacy_target_matcher", 
                                                           "medspacy_context", "medspacy_sectionizer"])
                
                # Add custom target rules for common medical entities in sepsis contexts
                from medspacy.target_matcher import TargetRule
                target_rules = [
                    TargetRule("sepsis", "CONDITION"),
                    TargetRule("bacteremia", "CONDITION"),
                    TargetRule("infection", "CONDITION"),
                    TargetRule("pneumonia", "CONDITION"),
                    TargetRule("antibiotic", "TREATMENT"),
                    TargetRule("vancomycin", "MEDICATION"),
                    TargetRule("piperacillin", "MEDICATION"),
                    TargetRule("tazobactam", "MEDICATION"),
                    TargetRule("meropenem", "MEDICATION"),
                    TargetRule("ceftriaxone", "MEDICATION"),
                    TargetRule("fever", "SYMPTOM"),
                    TargetRule("hypotension", "SYMPTOM"),
                    TargetRule("tachycardia", "SYMPTOM"),
                    TargetRule("leukocytosis", "LAB_RESULT"),
                    TargetRule("blood culture", "DIAGNOSTIC_PROCEDURE"),
                ]
                
                # Add rules to the target matcher
                target_matcher = self.medspacy_model.get_pipe("medspacy_target_matcher")
                for rule in target_rules:
                    target_matcher.add(rule)
                
                self.logger.info("MedSpaCy model initialized successfully with custom rules")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MedSpaCy: {str(e)}")
                self.logger.warning("MedSpaCy processing will be skipped.")
                self.args.use_medspacy = False
        elif self.args.use_medspacy and not HAS_MEDSPACY:
            self.logger.warning("MedSpaCy requested but not installed. Run: pip install medspacy")
            self.args.use_medspacy = False

        # Load specialized model for relationship extraction if needed
        if self.args.extract_relationships:
            try:
                self.logger.info("Loading en_core_sci_md model for dependency parsing...")
                self.sci_dep_model = spacy.load("en_core_sci_md")
                self.logger.info("Scientific model with dependency parsing loaded successfully")
            except OSError as e:
                self.logger.warning(f"Failed to load 'en_core_sci_md': {str(e)}")
                self.logger.warning("Using SciSpacy model for dependency parsing instead.")
                self.sci_dep_model = self.scispacy_model

        # Initialize negation detector if available
        if HAS_PYCONTEXT and (self.args.extract_relationships or self.args.visualize_entities):
            try:
                self.negex = PyConText()
                self.logger.info("PyConText negation detector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PyConText: {str(e)}")
                self.negex = None
        else:
            self.negex = None

    def load_data(self) -> pd.DataFrame:
        self.logger.info(f"Loading data from {self.args.db_path}")
        try:
            db_file = Path(self.args.db_path)
            if not db_file.exists():
                sys.exit(f"Database file not found: {self.args.db_path}")
            con = duckdb.connect(str(db_file))
            try:
                # Build condition for ICD-9 codes
                codes = [code.strip() for code in self.args.icd9_codes.split(',')]
                conditions = []
                for code in codes:
                    if code == "038":
                        conditions.append("ICD9_CODE LIKE '038%'")
                    else:
                        conditions.append(f"ICD9_CODE = '{code}'")
                icd_condition = " OR ".join(conditions)

                diagnoses_query = f"""
                SELECT DISTINCT SUBJECT_ID, HADM_ID, ICD9_CODE
                FROM "main"."DIAGNOSES_ICD"
                WHERE {icd_condition}
                """
                diag_df = con.execute(diagnoses_query).fetchdf()
                if diag_df.empty:
                    sys.exit(f"No patients found with ICD-9 codes: {self.args.icd9_codes}")

                notes_query = f"""
                WITH relevant_patients AS (
                    SELECT DISTINCT SUBJECT_ID
                    FROM "main"."DIAGNOSES_ICD"
                    WHERE {icd_condition}
                )
                SELECT n.SUBJECT_ID, n.HADM_ID, n.CHARTDATE, n.CATEGORY, n.TEXT
                FROM "main"."NOTEEVENTS" n
                JOIN relevant_patients r ON n.SUBJECT_ID = r.SUBJECT_ID
                WHERE n.CATEGORY = '{self.args.note_category}'
                  AND n.TEXT IS NOT NULL
                """
                if self.args.max_notes:
                    notes_query += f" LIMIT {self.args.max_notes}"
                notes_df = con.execute(notes_query).fetchdf()
                notes_df.columns = map(str.lower, notes_df.columns)
                self._validate_data(notes_df)
                self.logger.info(f"Loaded {len(notes_df)} notes from the database.")
                return notes_df
            except Exception as e:
                sys.exit(f"Database query failed: {str(e)}")
            finally:
                con.close()
        except Exception as e:
            sys.exit(f"Database operation failed: {str(e)}")

    def _validate_data(self, df: pd.DataFrame) -> None:
        try:
            if df.empty:
                sys.exit("Data validation failed: No notes found.")
            null_counts = df.isnull().sum()
            if null_counts.any():
                self.logger.warning(f"Null values in columns: {list(null_counts[null_counts > 0].index)}")
            text_lengths = df['text'].str.len()
            if text_lengths.min() < 10:
                self.logger.warning("Some texts are shorter than 10 characters; consider filtering them out.")
            self.logger.info(f"Text lengths: min={text_lengths.min()}, max={text_lengths.max()}, mean={text_lengths.mean():.2f}")
            short_texts = len(df[text_lengths < 100])
            if short_texts > 0:
                self.logger.warning(f"Found {short_texts} texts with length < 100 characters")
        except Exception as e:
            sys.exit(f"Data validation failed: {str(e)}")

    def process_documents(self, texts: List[str], batch_size: int = GLOBAL_BATCH_SIZE) -> Dict:
        """
        Process documents with SpaCy, SciSpacy, and MedSpaCy (if enabled), extracting entities.
        Now also returns the processed documents for visualization.
        """
        results = {
            'spacy_entities': [],
            'scispacy_entities': [],
            'clean_texts': [],
            'spacy_docs': [],   # Store docs for visualization
            'scispacy_docs': [] # Store docs for visualization
        }
        
        # Add MedSpaCy results if enabled
        if self.args.use_medspacy and self.medspacy_model:
            results['medspacy_entities'] = []
            results['medspacy_contexts'] = []
            results['medspacy_sections'] = []
            results['medspacy_docs'] = []
        
        total_batches = (len(texts) + batch_size - 1) // batch_size
        self.logger.info(f"Processing {len(texts)} documents in {total_batches} batches (batch size = {batch_size}).")
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing documents", total=total_batches):
            batch = texts[i:i + batch_size]
            try:
                # Process with SpaCy
                spacy_docs = list(self.spacy_model.pipe(batch))
                spacy_entities_batch = [[(ent.text, ent.label_) for ent in doc.ents] for doc in spacy_docs]
                
                # Process with SciSpaCy
                scispacy_docs = list(self.scispacy_model.pipe(batch))
                scispacy_entities_batch = [[(ent.text, ent.label_) for ent in doc.ents] for doc in scispacy_docs]
                
                # Generate clean text
                clean_text_batch = [self.preprocess_text(txt) for txt in batch]
                
                # Add to results
                results['spacy_entities'].extend(spacy_entities_batch)
                results['scispacy_entities'].extend(scispacy_entities_batch)
                results['clean_texts'].extend(clean_text_batch)
                results['spacy_docs'].extend(spacy_docs)
                results['scispacy_docs'].extend(scispacy_docs)
                
                # Process with MedSpaCy if enabled
                if self.args.use_medspacy and self.medspacy_model:
                    medspacy_docs = list(self.medspacy_model.pipe(batch))
                    
                    # Extract entities (targets) from MedSpaCy
                    medspacy_entities_batch = [
                        [(ent.text, ent.label_) for ent in doc.ents] for doc in medspacy_docs
                    ]
                    
                    # Extract context attributes (negation, hypothetical, historical, etc.)
                    medspacy_contexts_batch = [
                        [(ent.text, ent.label_, ent._.is_negated, ent._.is_historical, ent._.is_hypothetical) 
                         for ent in doc.ents] 
                        for doc in medspacy_docs
                    ]
                    
                    # Extract section information if available
                    medspacy_sections_batch = [
                        [(section.section_title, section.section_span.start_char, section.section_span.end_char)
                         for section in doc._.sections] 
                        for doc in medspacy_docs
                    ]
                    
                    # Add MedSpaCy results
                    results['medspacy_entities'].extend(medspacy_entities_batch)
                    results['medspacy_contexts'].extend(medspacy_contexts_batch)
                    results['medspacy_sections'].extend(medspacy_sections_batch)
                    results['medspacy_docs'].extend(medspacy_docs)
                
                # Memory cleanup
                if i % (batch_size * 5) == 0:
                    gc.collect()
                    if torch.cuda.is_available() and not self.args.disable_gpu:
                        torch.cuda.empty_cache()
            except Exception as e:
                self.logger.error(f"Error processing batch {i}: {str(e)}")
                continue
        
        return results

    def preprocess_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def analyze_entity_overlap(self, notes_df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Analyzing entity overlap between NLP models...")
        summary = []
        batch_size = 100
        for start_idx in tqdm(range(0, len(notes_df), batch_size), desc="Analyzing entity overlap"):
            end_idx = min(start_idx + batch_size, len(notes_df))
            batch_df = notes_df.iloc[start_idx:end_idx]
            for idx, row in batch_df.iterrows():
                spacy_set = set(ent[0].lower() for ent in row['spacy_entities'])
                scispacy_set = set(ent[0].lower() for ent in row['scispacy_entities'])
                
                # Include MedSpaCy if available
                if self.args.use_medspacy and 'medspacy_entities' in row:
                    medspacy_set = set(ent[0].lower() for ent in row['medspacy_entities'])
                    overlap_all = spacy_set.intersection(scispacy_set).intersection(medspacy_set)
                    only_medspacy = medspacy_set - (spacy_set.union(scispacy_set))
                    
                    summary.append({
                        'subject_id': row['subject_id'],
                        'hadm_id': row['hadm_id'],
                        'spacy_count': len(spacy_set),
                        'scispacy_count': len(scispacy_set),
                        'medspacy_count': len(medspacy_set),
                        'overlap_count': len(spacy_set.intersection(scispacy_set)),
                        'overlap_all_models': len(overlap_all),
                        'unique_spacy': len(spacy_set - scispacy_set - medspacy_set),
                        'unique_scispacy': len(scispacy_set - spacy_set - medspacy_set),
                        'unique_medspacy': len(only_medspacy)
                    })
                else:
                    # Original overlap analysis without MedSpaCy
                    overlap = spacy_set.intersection(scispacy_set)
                    summary.append({
                        'subject_id': row['subject_id'],
                        'hadm_id': row['hadm_id'],
                        'spacy_count': len(spacy_set),
                        'scispacy_count': len(scispacy_set),
                        'overlap_count': len(overlap),
                        'unique_spacy': len(spacy_set - scispacy_set),
                        'unique_scispacy': len(scispacy_set - spacy_set)
                    })
        return pd.DataFrame(summary)

    def train_word2vec(self, scispacy_entities: List[List[Tuple[str, str]]],
                       vector_size: int = 100, window: int = 5, min_count: int = 2,
                       sg: int = 1, epochs: int = 5) -> Word2Vec:
        self.logger.info("Training custom Word2Vec on SciSpacy entity tokens...")
        all_tokens = []
        for ents in tqdm(scispacy_entities, desc="Preparing entity tokens"):
            tokens = [token.lower() for token, _ in ents]
            all_tokens.append(tokens)
        if len(all_tokens) < 5:
            self.logger.warning("Not enough data for a meaningful Word2Vec model. Returning a minimal model.")
            return Word2Vec(min_count=1)
        total_tokens = sum(len(tokens) for tokens in all_tokens)
        self.logger.info(f"Training Word2Vec on {len(all_tokens)} documents with {total_tokens} tokens")
        model = Word2Vec(
            sentences=all_tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=sg,
            workers=4
        )
        model.train(all_tokens, total_examples=len(all_tokens), epochs=epochs)
        self.logger.info(f"Custom Word2Vec vocabulary size: {len(model.wv.key_to_index)}")
        return model

    def load_pretrained_embeddings(self, pretrained_path: str) -> KeyedVectors:
        """
        Load pre-trained word embeddings either from a local file or using gensim's downloader API.
        
        Args:
            pretrained_path: Path to local file or name of a pre-trained model available in gensim's API
                            (e.g., 'glove-wiki-gigaword-50', 'word2vec-google-news-300')
        
        Returns:
            KeyedVectors object containing the word embeddings
        """
        self.logger.info(f"Loading pre-trained embeddings: {pretrained_path}")
        
        # First, check if this is a model name from gensim's downloader
        try:
            # Get list of available models
            available_models = list(gensim_downloader.info()['models'].keys())
            
            if pretrained_path in available_models:
                self.logger.info(f"Downloading model '{pretrained_path}' using gensim downloader...")
                pretrained_model = gensim_downloader.load(pretrained_path)
                self.logger.info(f"Pre-trained embeddings loaded from gensim. Vocabulary size: {len(pretrained_model.key_to_index)}")
                return pretrained_model
        except Exception as e:
            self.logger.warning(f"Error checking gensim models: {str(e)}")
        
        # If not found in gensim or if there was an error, try loading from local file
        try:
            if os.path.exists(pretrained_path):
                # Determine if binary based on file extension
                binary = pretrained_path.endswith('.bin')
                pretrained_model = KeyedVectors.load_word2vec_format(pretrained_path, binary=binary)
                self.logger.info(f"Pre-trained embeddings loaded from file. Vocabulary size: {len(pretrained_model.key_to_index)}")
                return pretrained_model
            else:
                raise FileNotFoundError(f"File not found: {pretrained_path}")
        except Exception as e:
            self.logger.error(f"Failed to load pre-trained embeddings from file: {str(e)}")
            
        # If we reach here, attempt to download a default medical model
        self.logger.info("Attempting to load a default pre-trained model...")
        try:
            # First preference: BioWordVec if available
            if 'BioWordVec' in available_models:
                pretrained_model = gensim_downloader.load('BioWordVec')
            # Second preference: Any medical/bio model
            elif any(m for m in available_models if 'bio' in m.lower() or 'med' in m.lower()):
                med_model = next(m for m in available_models if 'bio' in m.lower() or 'med' in m.lower())
                pretrained_model = gensim_downloader.load(med_model)
            # Fallback: Use GloVe
            else:
                self.logger.warning("No medical embeddings found. Falling back to GloVe.")
                pretrained_model = gensim_downloader.load('glove-wiki-gigaword-50')
            
            self.logger.info(f"Loaded default pre-trained model. Vocabulary size: {len(pretrained_model.key_to_index)}")
            return pretrained_model
        except Exception as e:
            self.logger.error(f"Failed to load any pre-trained embeddings: {str(e)}")
            raise ValueError("Could not load pre-trained embeddings from any source.")

    def visualize_medspacy_sections(self, docs: List[Any], num_examples: int = 3, 
                                   output_dir: Optional[str] = None) -> None:
        """
        Create visualizations of document sections identified by MedSpaCy.
        
        Args:
            docs: List of processed MedSpaCy documents
            num_examples: Number of documents to visualize
            output_dir: Directory to save HTML files
        """
        if not docs:
            self.logger.warning("No MedSpaCy documents to visualize.")
            return
            
        self.logger.info(f"Creating section visualizations for {min(num_examples, len(docs))} documents")
        
        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir) / "medspacy_sections"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample documents
        if len(docs) > num_examples:
            sample_indices = random.sample(range(len(docs)), num_examples)
            sample_docs = [docs[i] for i in sample_indices]
        else:
            sample_docs = docs[:num_examples]
        
        # Visualize sections for each document
        for i, doc in enumerate(sample_docs):
            try:
                # Get sections
                if not hasattr(doc._, "sections") or not doc._.sections:
                    continue
                    
                # Create HTML visualization
                html = ["<!DOCTYPE html>", "<html>", "<head>", 
                       "<title>MedSpaCy Sections</title>", 
                       "<style>",
                       "body { font-family: Arial, sans-serif; margin: 20px; }",
                       ".section { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }",
                       ".section-title { font-weight: bold; color: #2c3e50; }",
                       ".section-content { margin-top: 5px; }",
                       ".entity { display: inline-block; border-radius: 3px; padding: 0 5px; margin: 0 2px; }",
                       ".CONDITION { background-color: #ffa07a; }",
                       ".TREATMENT { background-color: #90ee90; }",
                       ".MEDICATION { background-color: #add8e6; }",
                       ".SYMPTOM { background-color: #ffb6c1; }",
                       ".LAB_RESULT { background-color: #e6e6fa; }",
                       ".DIAGNOSTIC_PROCEDURE { background-color: #f0e68c; }",
                       "</style>",
                       "</head>",
                       "<body>",
                       f"<h1>Document {i+1} Sections</h1>"]
                
                # Add each section
                for section in doc._.sections:
                    section_text = doc.text[section.section_span.start_char:section.section_span.end_char]
                    
                    # Format section with title
                    html.append(f'<div class="section">')
                    html.append(f'<div class="section-title">{section.section_title or "Unnamed Section"}</div>')
                    
                    # Handle entities in this section
                    section_content = section_text
                    section_entities = []
                    
                    # Find entities in this section
                    for ent in doc.ents:
                        if (ent.start_char >= section.section_span.start_char and 
                            ent.end_char <= section.section_span.end_char):
                            section_entities.append(ent)
                    
                    # Sort entities by position
                    section_entities.sort(key=lambda x: x.start_char)
                    
                    # Format text with entity markup
                    if section_entities:
                        formatted_text = []
                        last_end = section.section_span.start_char
                        
                        for ent in section_entities:
                            # Add text before entity
                            if ent.start_char > last_end:
                                formatted_text.append(doc.text[last_end:ent.start_char])
                            
                            # Add entity with markup
                            formatted_text.append(f'<span class="entity {ent.label_}">{ent.text}</span>')
                            last_end = ent.end_char
                        
                        # Add remaining text
                        if last_end < section.section_span.end_char:
                            formatted_text.append(doc.text[last_end:section.section_span.end_char])
                        
                        section_content = "".join(formatted_text)
                    
                    html.append(f'<div class="section-content">{section_content}</div>')
                    html.append('</div>')
                
                html.extend(["</body>", "</html>"])
                
                # Write to file
                if output_dir:
                    output_file = output_path / f"section_doc_{i}.html"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(html))
                    self.logger.info(f"Section visualization saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error generating section visualization for document {i}: {str(e)}")

    def visualize_medspacy_context(self, docs: List[Any], num_examples: int = 5,
                                 output_dir: Optional[str] = None) -> None:
        """
        Create visualizations of MedSpaCy context attributes (negation, historical, hypothetical).
        
        Args:
            docs: List of processed MedSpaCy documents
            num_examples: Number of documents to visualize
            output_dir: Directory to save HTML files
        """
        if not docs:
            self.logger.warning("No MedSpaCy documents to visualize.")
            return
            
        self.logger.info(f"Creating context visualizations for {min(num_examples, len(docs))} documents")
        
        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir) / "medspacy_context"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample documents
        if len(docs) > num_examples:
            sample_indices = random.sample(range(len(docs)), num_examples)
            sample_docs = [docs[i] for i in sample_indices]
        else:
            sample_docs = docs[:num_examples]
        
        # Visualize context for each document
        for i, doc in enumerate(sample_docs):
            try:
                # Skip if no entities
                if not doc.ents:
                    continue
                    
                # Create HTML visualization
                html = ["<!DOCTYPE html>", "<html>", "<head>", 
                       "<title>MedSpaCy Context</title>", 
                       "<style>",
                       "body { font-family: Arial, sans-serif; margin: 20px; }",
                       "table { border-collapse: collapse; width: 100%; }",
                       "th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }",
                       "th { background-color: #f2f2f2; }",
                       "tr:nth-child(even) { background-color: #f9f9f9; }",
                       ".negated { color: red; }",
                       ".historical { color: blue; }",
                       ".hypothetical { color: purple; }",
                       "</style>",
                       "</head>",
                       "<body>",
                       f"<h1>Document {i+1} Entity Context</h1>",
                       "<table>",
                       "<tr><th>Entity</th><th>Type</th><th>Negated</th><th>Historical</th><th>Hypothetical</th><th>Context</th></tr>"]
                
                # Add each entity with context
                for ent in doc.ents:
                    # Get context span (words around entity)
                    start_idx = max(0, ent.start_char - 40)
                    end_idx = min(len(doc.text), ent.end_char + 40)
                    context_text = doc.text[start_idx:end_idx]
                    
                    # Replace entity in context with styled version
                    entity_class = []
                    if hasattr(ent._, "is_negated") and ent._.is_negated:
                        entity_class.append("negated")
                    if hasattr(ent._, "is_historical") and ent._.is_historical:
                        entity_class.append("historical")
                    if hasattr(ent._, "is_hypothetical") and ent._.is_hypothetical:
                        entity_class.append("hypothetical")
                    
                    entity_span = f'<span class="{" ".join(entity_class)}">{ent.text}</span>'
                    
                    # Get context with highlighted entity
                    pre_context = context_text[:ent.start_char - start_idx]
                    post_context = context_text[ent.end_char - start_idx:]
                    highlighted_context = f"{pre_context}{entity_span}{post_context}"
                    
                    # Add row to table
                    is_negated = "Yes" if hasattr(ent._, "is_negated") and ent._.is_negated else "No"
                    is_historical = "Yes" if hasattr(ent._, "is_historical") and ent._.is_historical else "No"
                    is_hypothetical = "Yes" if hasattr(ent._, "is_hypothetical") and ent._.is_hypothetical else "No"
                    
                    html.append("<tr>")
                    html.append(f"<td>{ent.text}</td>")
                    html.append(f"<td>{ent.label_}</td>")
                    html.append(f"<td>{is_negated}</td>")
                    html.append(f"<td>{is_historical}</td>")
                    html.append(f"<td>{is_hypothetical}</td>")
                    html.append(f"<td>{highlighted_context}</td>")
                    html.append("</tr>")
                
                html.extend(["</table>", "</body>", "</html>"])
                
                # Write to file
                if output_dir:
                    output_file = output_path / f"context_doc_{i}.html"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write("\n".join(html))
                    self.logger.info(f"Context visualization saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error generating context visualization for document {i}: {str(e)}")

    def analyze_medspacy_findings(self, docs: List[Any], output_dir: Optional[str] = None) -> Dict:
        """
        Analyze and visualize MedSpaCy findings including context attributes and sections.
        """
        if not docs:
            self.logger.warning("No MedSpaCy documents to analyze.")
            return {}
            
        self.logger.info(f"Analyzing MedSpaCy findings for {len(docs)} documents")
        
        # Initialize counters
        entity_types = Counter()
        entity_counts = Counter()
        section_counts = Counter()
        context_stats = {
            'negated': Counter(),
            'historical': Counter(),
            'hypothetical': Counter()
        }
        
        # Process all documents
        for doc in tqdm(docs, desc="Analyzing MedSpaCy findings"):
            # Count entity types
            for ent in doc.ents:
                entity_types[ent.label_] += 1
                entity_counts[ent.text.lower()] += 1
                
                # Track context attributes
                if hasattr(ent._, "is_negated") and ent._.is_negated:
                    context_stats['negated'][ent.text.lower()] += 1
                if hasattr(ent._, "is_historical") and ent._.is_historical:
                    context_stats['historical'][ent.text.lower()] += 1
                if hasattr(ent._, "is_hypothetical") and ent._.is_hypothetical:
                    context_stats['hypothetical'][ent.text.lower()] += 1
            
            # Count sections if available
            if hasattr(doc._, "sections"):
                for section in doc._.sections:
                    section_title = section.section_title or "Unknown Section"
                    section_counts[section_title] += 1
        
        # Skip visualizations if no entities found
        if not entity_types:
            self.logger.warning("No entities found for visualization.")
            return {
                'entity_types': entity_types,
                'entity_counts': entity_counts,
                'section_counts': section_counts,
                'context_stats': context_stats
            }
        
        # Create summary visualizations if output directory provided
        if output_dir and HAS_SNS:
            output_path = Path(output_dir) / "medspacy_analysis"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Entity type distribution (only if entities exist)
            if entity_types:
                plt.figure(figsize=(10, 6))
                sns.barplot(x=list(entity_types.keys()), y=list(entity_types.values()))
                plt.title("MedSpaCy Entity Type Distribution")
                plt.xlabel("Entity Type")
                plt.ylabel("Count")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.savefig(output_path / "entity_type_distribution.png", dpi=300)
                plt.close()
            
            # Top entities by frequency (only if entities exist)
            if entity_counts:
                top_entities = entity_counts.most_common(20)
                if top_entities:  # Only create visualization if there are entities
                    plt.figure(figsize=(12, 8))
                    sns.barplot(x=[e[1] for e in top_entities], y=[e[0] for e in top_entities])
                    plt.title("Top 20 Entities by Frequency")
                    plt.xlabel("Count")
                    plt.tight_layout()
                    plt.savefig(output_path / "top_entities.png", dpi=300)
                    plt.close()
            
            # Context attribute analysis
            # Check if there are any context attributes to visualize
            if (sum(context_stats['negated'].values()) > 0 or 
                sum(context_stats['historical'].values()) > 0 or 
                sum(context_stats['hypothetical'].values()) > 0):
                
                plt.figure(figsize=(12, 8))
                context_data = {
                    'Negated': sum(context_stats['negated'].values()),
                    'Historical': sum(context_stats['historical'].values()),
                    'Hypothetical': sum(context_stats['hypothetical'].values()),
                    'Standard': sum(entity_types.values()) - sum(context_stats['negated'].values()) -
                                sum(context_stats['historical'].values()) - sum(context_stats['hypothetical'].values())
                }
                sns.barplot(x=list(context_data.keys()), y=list(context_data.values()))
                plt.title("Entity Context Attribute Distribution")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(output_path / "context_distribution.png", dpi=300)
                plt.close()
            
            # Section distribution if available
            if section_counts:
                top_sections = section_counts.most_common(15)
                if top_sections:  # Only create visualization if there are sections
                    plt.figure(figsize=(14, 8))
                    sns.barplot(x=[s[1] for s in top_sections], y=[s[0] for s in top_sections])
                    plt.title("Top 15 Document Sections")
                    plt.xlabel("Count")
                    plt.tight_layout()
                    plt.savefig(output_path / "section_distribution.png", dpi=300)
                    plt.close()
        
        # Return analysis results
        return {
            'entity_types': entity_types,
            'entity_counts': entity_counts,
            'section_counts': section_counts,
            'context_stats': context_stats
        }

    def compare_entity_recognition(self, notes_df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
        """
        Compare entity recognition capabilities of SpaCy, SciSpaCy and MedSpaCy.
        
        Args:
            notes_df: DataFrame with processed notes
            output_dir: Directory to save visualizations
        """
        if not self.args.use_medspacy or 'medspacy_entities' not in notes_df.columns:
            self.logger.warning("MedSpaCy results not available for comparison.")
            return
            
        self.logger.info("Comparing entity recognition between NLP models")
        
        # Extract entity sets by model
        entity_sets = {
            'spacy': set(),
            'scispacy': set(),
            'medspacy': set()
        }
        
        entity_types = {
            'spacy': Counter(),
            'scispacy': Counter(),
            'medspacy': Counter()
        }
        
        # Process all notes
        for _, row in notes_df.iterrows():
            for ent, label in row['spacy_entities']:
                entity_sets['spacy'].add(ent.lower())
                entity_types['spacy'][label] += 1
            
            for ent, label in row['scispacy_entities']:
                entity_sets['scispacy'].add(ent.lower())
                entity_types['scispacy'][label] += 1
            
            for ent, label in row['medspacy_entities']:
                entity_sets['medspacy'].add(ent.lower())
                entity_types['medspacy'][label] += 1
        
        # Calculate overlaps
        spacy_scispacy = entity_sets['spacy'].intersection(entity_sets['scispacy'])
        spacy_medspacy = entity_sets['spacy'].intersection(entity_sets['medspacy'])
        scispacy_medspacy = entity_sets['scispacy'].intersection(entity_sets['medspacy'])
        all_models = entity_sets['spacy'].intersection(entity_sets['scispacy']).intersection(entity_sets['medspacy'])
        
        # Log comparison results
        self.logger.info(f"Unique entities by model:")
        self.logger.info(f"  SpaCy: {len(entity_sets['spacy'])}")
        self.logger.info(f"  SciSpaCy: {len(entity_sets['scispacy'])}")
        self.logger.info(f"  MedSpaCy: {len(entity_sets['medspacy'])}")
        self.logger.info(f"Entity overlap:")
        self.logger.info(f"  SpaCy-SciSpaCy: {len(spacy_scispacy)}")
        self.logger.info(f"  SpaCy-MedSpaCy: {len(spacy_medspacy)}")
        self.logger.info(f"  SciSpaCy-MedSpaCy: {len(scispacy_medspacy)}")
        self.logger.info(f"  All models: {len(all_models)}")
        
        # Create visualizations if output directory provided
        if output_dir and HAS_SNS:
            from matplotlib_venn import venn3
            
            output_path = Path(output_dir) / "model_comparison"
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Venn diagram of entity overlap
            plt.figure(figsize=(10, 10))
            venn3([entity_sets['spacy'], entity_sets['scispacy'], entity_sets['medspacy']], 
                  ('SpaCy', 'SciSpaCy', 'MedSpaCy'))
            plt.title("Entity Overlap Between NLP Models")
            plt.tight_layout()
            plt.savefig(output_path / "entity_overlap_venn.png", dpi=300)
            plt.close()
            
            # Entity type distribution by model
            plt.figure(figsize=(15, 10))
            
            # Get common entity types
            all_types = set()
            for model in entity_types:
                all_types.update(entity_types[model].keys())
            
            # Prepare data for grouped bar chart
            model_names = list(entity_types.keys())
            type_names = sorted(all_types)
            data = []
            
            for model in model_names:
                model_data = []
                for type_name in type_names:
                    model_data.append(entity_types[model][type_name])
                data.append(model_data)
            
            # Plot grouped bar chart
            x = np.arange(len(type_names))
            width = 0.25
            
            fig, ax = plt.subplots(figsize=(15, 10))
            for i, model in enumerate(model_names):
                ax.bar(x + i*width - width, data[i], width, label=model)
            
            ax.set_xticks(x)
            ax.set_xticklabels(type_names, rotation=45, ha='right')
            ax.set_title('Entity Type Distribution by Model')
            ax.set_ylabel('Count')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / "entity_type_comparison.png", dpi=300)
            plt.close()
        
        return {
            'entity_sets': entity_sets,
            'entity_types': entity_types,
            'overlaps': {
                'spacy_scispacy': spacy_scispacy,
                'spacy_medspacy': spacy_medspacy,
                'scispacy_medspacy': scispacy_medspacy,
                'all_models': all_models
            }
        }

    def visualize_entities_with_displacy(self, docs: List[Any], num_examples: int = 5, 
                                        output_dir: Optional[str] = None) -> None:
        """
        Create visualizations of named entities using displaCy.
        
        Args:
            docs: List of processed spaCy documents
            num_examples: Number of documents to visualize
            output_dir: Directory to save HTML files
        """
        from spacy import displacy
        
        self.logger.info(f"Creating entity visualizations for {min(num_examples, len(docs))} documents")
        
        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir) / "entity_viz"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample a subset of documents if there are many
        if len(docs) > num_examples:
            sample_indices = random.sample(range(len(docs)), num_examples)
            sample_docs = [docs[i] for i in sample_indices]
        else:
            sample_docs = docs
        
        # Generate visualizations for each document
        for i, doc in enumerate(sample_docs):
            # Set custom colors for medical entity types
            colors = {
                "CHEMICAL": "#FF9561", 
                "DISEASE": "#DC4343", 
                "DATE": "#7AECEC",
                "PERSON": "#aa9cfc",
                "ORG": "#7aecec",
                "GPE": "#feca74",
                "LOC": "#9cc9cc",
                "CONDITION": "#ff8c69",
                "TREATMENT": "#90ee90",
                "MEDICATION": "#add8e6",
                "SYMPTOM": "#ffb6c1",
                "LAB_RESULT": "#e6e6fa",
                "DIAGNOSTIC_PROCEDURE": "#f0e68c"
            }
            
            # Get entity types in this doc
            entity_types = set(ent.label_ for ent in doc.ents)
            options = {"ents": list(entity_types), "colors": colors}
            
            # Generate HTML
            try:
                html = displacy.render(doc, style="ent", options=options, page=True)
                
                # Save to file if output directory specified
                if output_dir:
                    output_file = output_path / f"entity_doc_{i}.html"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(html)
                    self.logger.info(f"Entity visualization saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error generating displaCy visualization for document {i}: {str(e)}")

    def create_entity_summary_visualization(self, entities_by_type: Dict[str, Counter], 
                                           output_dir: Optional[str] = None) -> None:
        """
        Create a summary visualization of the most common entities by type.
        
        Args:
            entities_by_type: Dictionary mapping entity types to Counter objects
            output_dir: Directory to save visualization
        """
        if not HAS_SNS:
            self.logger.warning("Seaborn not installed; skipping summary visualization.")
            return
            
        import seaborn as sns
        from matplotlib.gridspec import GridSpec
        
        # Set up the figure
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Get the top entity types by count
        entity_types = sorted(entities_by_type.keys(), 
                            key=lambda x: sum(entities_by_type[x].values()), 
                            reverse=True)[:4]
        
        # Plot each entity type
        for i, entity_type in enumerate(entity_types):
            ax = plt.subplot(gs[i // 2, i % 2])
            
            # Get the top 10 entities for this type
            top_entities = entities_by_type[entity_type].most_common(10)
            if not top_entities:
                continue
                
            labels, values = zip(*top_entities)
            
            # Create horizontal bar chart
            sns.barplot(x=list(values), y=list(labels), palette="viridis", ax=ax)
            ax.set_title(f"Top 10 {entity_type} Entities")
            ax.set_xlabel("Count")
        
        plt.tight_layout()
        
        # Save if output directory specified
        if output_dir:
            output_path = Path(output_dir) / "entity_summary.png"
            plt.savefig(output_path, bbox_inches="tight", dpi=300)
            self.logger.info(f"Entity summary visualization saved to {output_path}")
        
        try:
            plt.show()
        except Exception as e:
            self.logger.warning(f"Could not display plot interactively: {e}")

    def enhance_entity_extraction(self, texts: List[str], batch_size: int = 32) -> Dict:
        """
        Enhanced entity extraction with POS tagging, entity context, and negation detection.
        
        Args:
            texts: List of text documents
            batch_size: Batch size for processing
            
        Returns:
            Enhanced entity information including context and linguistic features
        """
        # Initialize negation detection if PyConText is available
        negex = self.negex if HAS_PYCONTEXT else None
        
        # Initialize enhanced results
        results = {
            'entities': [],
            'entity_contexts': [],
            'negated_entities': [],
            'pos_tags': []
        }
        
        # Process in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="Enhanced entity processing", total=total_batches):
            batch = texts[i:i + batch_size]
            
            # Process with SciSpacy
            docs = list(self.scispacy_model.pipe(batch))
            
            batch_entities = []
            batch_contexts = []
            batch_negated = []
            batch_pos = []
            
            for doc in docs:
                # Extract entities with linguistic features
                doc_entities = []
                doc_contexts = []
                doc_negated = []
                doc_pos = {}
                
                # Track POS tags
                for token in doc:
                    if token.pos_ not in doc_pos:
                        doc_pos[token.pos_] = []
                    doc_pos[token.pos_].append(token.text)
                
                # Process entities
                for ent in doc.ents:
                    # Get entity with POS and dependency information
                    entity_info = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'pos': ent.root.pos_,
                        'dep': ent.root.dep_,
                    }
                    doc_entities.append(entity_info)
                    
                    # Extract context (window of words around entity)
                    left_context = doc.text[max(0, ent.start_char-50):ent.start_char].strip()
                    right_context = doc.text[ent.end_char:min(len(doc.text), ent.end_char+50)].strip()
                    doc_contexts.append({
                        'entity': ent.text,
                        'left': left_context,
                        'right': right_context
                    })
                    
                    # Negation detection with PyConText if available
                    is_negated = False
                    if negex is not None and ent.label_ == "DISEASE":
                        try:
                            context_text = f"{left_context} {ent.text} {right_context}"
                            markup = negex.markup_sentence(context_text)
                            if markup.getTargets() and any(target.getCategory() == "NEGATED" for target in markup.getTargets()):
                                is_negated = True
                        except Exception as e:
                            self.logger.warning(f"Error in negation detection: {str(e)}")
                    
                    doc_negated.append({
                        'entity': ent.text,
                        'is_negated': is_negated
                    })
                
                batch_entities.append(doc_entities)
                batch_contexts.append(doc_contexts)
                batch_negated.append(doc_negated)
                batch_pos.append(doc_pos)
            
            results['entities'].extend(batch_entities)
            results['entity_contexts'].extend(batch_contexts)
            results['negated_entities'].extend(batch_negated)
            results['pos_tags'].extend(batch_pos)
            
            # Clear memory
            del docs
            if i % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available() and not self.args.disable_gpu:
                    torch.cuda.empty_cache()
        
        return results

    def analyze_entity_frequencies(self, entity_data: Dict) -> Dict[str, Counter]:
        """
        Analyze frequency distributions of entities by type.
        
        Args:
            entity_data: Enhanced entity information from enhance_entity_extraction
            
        Returns:
            Dictionary mapping entity types to Counter objects of entity frequencies
        """
        from collections import Counter
        
        # Initialize counters by entity type
        entities_by_type = {}
        
        # Process all documents
        for doc_entities in entity_data['entities']:
            for entity in doc_entities:
                entity_type = entity['label']
                entity_text = entity['text'].lower()
                
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = Counter()
                
                entities_by_type[entity_type][entity_text] += 1
        
        return entities_by_type

    def create_entity_co_occurrence_network(self, entity_data: Dict, min_co_occurrences: int = 2,
                                          output_dir: Optional[str] = None) -> Optional[Any]:
        """
        Create a network of co-occurring entities.
        
        Args:
            entity_data: Enhanced entity information
            min_co_occurrences: Minimum number of co-occurrences to include in network
            output_dir: Directory to save visualization
            
        Returns:
            NetworkX graph of entity co-occurrences or None if networkx not available
        """
        if not HAS_NX:
            self.logger.warning("NetworkX not installed; skipping co-occurrence network.")
            return None
            
        import networkx as nx
        from collections import Counter
        
        # Track co-occurrences
        co_occurrences = Counter()
        
        # Process all documents
        for doc_entities in entity_data['entities']:
            # Get unique entities in this document
            entities = [(e['text'].lower(), e['label']) for e in doc_entities]
            unique_entities = set(entities)
            
            # Count co-occurrences
            for i, (entity1, type1) in enumerate(unique_entities):
                for entity2, type2 in list(unique_entities)[i+1:]:
                    if type1 == type2:  # Only connect entities of same type
                        pair = tuple(sorted([entity1, entity2]))
                        co_occurrences[pair] += 1
        
        # Create network
        G = nx.Graph()
        
        # Add edges for co-occurrences that meet the threshold
        for (entity1, entity2), count in co_occurrences.items():
            if count >= min_co_occurrences:
                G.add_edge(entity1, entity2, weight=count)
        
        # Visualize if the graph has nodes
        if G.number_of_nodes() > 0:
            plt.figure(figsize=(12, 10))
            
            # Choose layout based on graph size
            if G.number_of_nodes() < 100:
                pos = nx.spring_layout(G, seed=42)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=300, alpha=0.7)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            
            # Add labels for the largest nodes
            degree = dict(nx.degree(G, weight='weight'))
            top_nodes = sorted(degree, key=degree.get, reverse=True)[:20]
            labels = {node: node for node in top_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
            
            plt.title("Entity Co-occurrence Network")
            plt.axis('off')
            
            # Save if output directory specified
            if output_dir:
                output_path = Path(output_dir) / "entity_cooccurrence_network.png"
                plt.savefig(output_path, bbox_inches="tight", dpi=300)
                self.logger.info(f"Entity co-occurrence network saved to {output_path}")
            
            try:
                plt.show()
            except Exception as e:
                self.logger.warning(f"Could not display plot interactively: {e}")
        else:
            self.logger.warning("No entity co-occurrences met the threshold. No network to visualize.")
        
        return G

    def extract_entity_relationships(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Extract relationships between entities using dependency parsing.
        
        Args:
            texts: List of text documents
            batch_size: Batch size for processing
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        # Process in batches
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting relationships", total=total_batches):
            batch = texts[i:i + batch_size]
            
            # Process texts with sci_dep_model (which has the dependency parser)
            docs = list(self.sci_dep_model.pipe(batch))
            
            # Extract relationships using patterns
            for doc in docs:
                # Extract sentences
                for sent in doc.sents:
                    # Look for verbs that might indicate relationships
                    for token in sent:
                        if token.pos_ == "VERB":
                            verb = token.text.lower()
                            
                            # Skip common non-informative verbs
                            if verb in ["is", "are", "was", "were", "be", "been"]:
                                continue
                                
                            # Find subject
                            subj = None
                            subj_ent = None
                            
                            for child in token.children:
                                if child.dep_ in ("nsubj", "nsubjpass"):
                                    subj = child
                                    # Check if subject is part of an entity
                                    for ent in doc.ents:
                                        if child.i >= ent.start and child.i < ent.end:
                                            subj = ent
                                            subj_ent = ent.label_
                                            break
                                    break
                            
                            # Find object
                            obj = None
                            obj_ent = None
                            
                            # First look for direct objects
                            for child in token.children:
                                if child.dep_ in ("dobj", "pobj", "attr"):
                                    obj = child
                                    # Check if object is part of an entity
                                    for ent in doc.ents:
                                        if child.i >= ent.start and child.i < ent.end:
                                            obj = ent
                                            obj_ent = ent.label_
                                            break
                                    break
                            
                            # If no direct object, look for prepositional objects
                            if obj is None:
                                for child in token.children:
                                    if child.dep_ == "prep":
                                        for grandchild in child.children:
                                            if grandchild.dep_ == "pobj":
                                                obj = grandchild
                                                # Check if object is part of an entity
                                                for ent in doc.ents:
                                                    if grandchild.i >= ent.start and grandchild.i < ent.end:
                                                        obj = ent
                                                        obj_ent = ent.label_
                                                        break
                                                break
                                        if obj is not None:
                                            break
                            
                            # Record relationship if both subject and object are found
                            if subj and obj:
                                relationship = {
                                    'subject': subj.text,
                                    'subject_type': subj_ent,
                                    'verb': token.text,
                                    'object': obj.text,
                                    'object_type': obj_ent,
                                    'sentence': sent.text
                                }
                                relationships.append(relationship)
            
            # Clear memory
            del docs
            if i % (batch_size * 5) == 0:
                gc.collect()
                if torch.cuda.is_available() and not self.args.disable_gpu:
                    torch.cuda.empty_cache()
        
        return relationships

    def visualize_dependency_relations(self, docs: List[Any], num_examples: int = 3, 
                                      output_dir: Optional[str] = None) -> None:
        """
        Visualize dependency relationships using displaCy.
        
        Args:
            docs: List of processed spaCy documents
            num_examples: Number of sentences to visualize
            output_dir: Directory to save HTML files
        """
        from spacy import displacy
        
        self.logger.info(f"Creating dependency visualizations for {num_examples} sentences")
        
        # Create output directory if needed
        if output_dir:
            output_path = Path(output_dir) / "dependency_viz"
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract sample sentences
        sentences = []
        for doc in docs:
            sentences.extend(list(doc.sents))
        
        # Sample a subset if there are many
        if len(sentences) > num_examples:
            sample_sentences = random.sample(sentences, num_examples)
        else:
            sample_sentences = sentences[:num_examples]
        
        # Generate visualizations
        for i, sent in enumerate(sample_sentences):
            # Generate HTML
            try:
                html = displacy.render(sent, style="dep", options={"compact": True, "distance": 120}, page=True)
                
                # Save to file if output directory specified
                if output_dir:
                    output_file = output_path / f"dependency_{i}.html"
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(html)
                    self.logger.info(f"Dependency visualization saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Error generating dependency visualization for sentence {i}: {str(e)}")

    def build_relationship_knowledge_graph(self, relationships: List[Dict], 
                                         output_dir: Optional[str] = None) -> Optional[Any]:
        """
        Build a knowledge graph from extracted relationships.
        
        Args:
            relationships: List of extracted relationships
            output_dir: Directory to save visualization
            
        Returns:
            NetworkX graph of relationships or None if networkx not available
        """
        if not HAS_NX:
            self.logger.warning("NetworkX not installed; skipping knowledge graph.")
            return None
            
        import networkx as nx
        
        # Create network
        G = nx.DiGraph()
        
        # Add edges for relationships between typed entities
        rel_count = 0
        for rel in relationships:
            if rel['subject_type'] and rel['object_type']:  # Only include typed entities
                G.add_edge(rel['subject'], rel['object'], 
                          type=rel['verb'], 
                          subject_type=rel['subject_type'],
                          object_type=rel['object_type'])
                rel_count += 1
        
        self.logger.info(f"Built knowledge graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        if G.number_of_nodes() > 0:
            # Visualize
            plt.figure(figsize=(12, 10))
            
            # Use different colors for different entity types
            node_colors = []
            node_types = {}
            
            # Find node types
            for s, t, data in G.edges(data=True):
                if s not in node_types and 'subject_type' in data:
                    node_types[s] = data['subject_type']
                if t not in node_types and 'object_type' in data:
                    node_types[t] = data['object_type']
            
            # Map types to colors
            color_map = {}
            all_types = set(node_types.values())
            for i, t in enumerate(all_types):
                color_map[t] = i
                
            # Assign colors
            for node in G.nodes():
                if node in node_types:
                    node_colors.append(color_map.get(node_types[node], 0))
                else:
                    node_colors.append(0)
            
            # Choose layout based on graph size
            if G.number_of_nodes() < 100:
                pos = nx.spring_layout(G, seed=42)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Draw network
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color=node_colors, 
                                  cmap=plt.cm.tab20, alpha=0.7)
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
            
            # Add labels for the largest nodes
            top_nodes = sorted(G.nodes(), key=G.degree, reverse=True)[:20]
            labels = {node: node for node in top_nodes}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
            
            plt.title("Knowledge Graph of Entity Relationships")
            plt.axis('off')
            
            # Create legend for entity types
            if color_map:
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                         markerfacecolor=plt.cm.tab20(color_map[t]), 
                                         label=t, markersize=10)
                                  for t in color_map]
                plt.legend(handles=legend_elements, loc='upper right')
            
            # Save if output directory specified
            if output_dir:
                output_path = Path(output_dir) / "relationship_knowledge_graph.png"
                plt.savefig(output_path, bbox_inches="tight", dpi=300)
                self.logger.info(f"Relationship knowledge graph saved to {output_path}")
            
            try:
                plt.show()
            except Exception as e:
                self.logger.warning(f"Could not display plot interactively: {e}")
        else:
            self.logger.warning("No entity relationships to visualize in knowledge graph.")
        
        return G

    def compare_embeddings(self, custom_model: Word2Vec, pretrained_model: KeyedVectors, query: str, topn: int = 10) -> None:
        self.logger.info(f"Comparing embeddings for query term: '{query}'")
        try:
            custom_similar = custom_model.wv.most_similar(query, topn=topn) if query in custom_model.wv.key_to_index else []
        except Exception as e:
            custom_similar = []
            self.logger.warning(f"Custom model does not contain '{query}': {e}")
        try:
            pretrained_similar = pretrained_model.most_similar(query, topn=topn) if query in pretrained_model.key_to_index else []
        except Exception as e:
            pretrained_similar = []
            self.logger.warning(f"Pre-trained model does not contain '{query}': {e}")
        self.logger.info("Custom model similar words:")
        for word, score in custom_similar:
            self.logger.info(f"  {word}: {score:.4f}")
        self.logger.info("Pre-trained model similar words:")
        for word, score in pretrained_similar:
            self.logger.info(f"  {word}: {score:.4f}")

    def visualize_embeddings_multiple(self, model: Union[Word2Vec, KeyedVectors], topn: int = 200) -> None:
        """
        Create multiple dimensionality reduction visualizations for word embeddings.
        
        Args:
            model: Word2Vec model or KeyedVectors
            topn: Number of top tokens to include
        """
        self.logger.info("Creating multiple dimensionality reduction visualizations...")
        vocab = list(model.wv.index_to_key) if isinstance(model, Word2Vec) else list(model.key_to_index)
        if not vocab:
            self.logger.warning("No vocabulary found; skipping visualization.")
            return
        vocab = vocab[:topn]
        vectors = np.array([model.wv[word] for word in vocab]) if isinstance(model, Word2Vec) else np.array([model[word] for word in vocab])
        methods = {}
        # TSNE
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vocab)-1))
        methods['TSNE'] = tsne_model.fit_transform(vectors)
        # PCA
        pca_model = PCA(n_components=2, random_state=42)
        methods['PCA'] = pca_model.fit_transform(vectors)
        # UMAP, if available
        if HAS_UMAP:
            umap_model = umap.UMAP(n_components=2, random_state=42)
            methods['UMAP'] = umap_model.fit_transform(vectors)
        # Plot side-by-side
        num_plots = len(methods)
        fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 6))
        if num_plots == 1:
            axes = [axes]
        for ax, (name, reduced) in zip(axes, methods.items()):
            ax.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
            for i, word in enumerate(vocab[:20]):  # label top 20 for clarity
                ax.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)
            ax.set_title(f"{name} Visualization")
        plt.tight_layout()
        output_dir = Path(self.args.output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / "dimensionality_reduction_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        self.logger.info(f"Dimensionality reduction comparison saved to {plot_path}")
        try:
            plt.show()
        except Exception as e:
            self.logger.warning(f"Could not display plots interactively: {e}")

    def semantic_similarity_analysis(self, model: Union[Word2Vec, KeyedVectors], query: str, topn: int = 10) -> None:
        self.logger.info(f"Performing semantic similarity analysis for query '{query}'")
        try:
            similar = model.wv.most_similar(query, topn=topn) if isinstance(model, Word2Vec) else model.most_similar(query, topn=topn)
        except Exception as e:
            self.logger.error(f"Error finding similar words for '{query}': {e}")
            return
        self.logger.info("Similar words:")
        for word, score in similar:
            self.logger.info(f"  {word}: {score:.4f}")
        if HAS_NX:
            G = nx.Graph()
            G.add_node(query)
            for word, score in similar:
                G.add_edge(query, word, weight=score)
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(8, 6))
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
            nx.draw_networkx_edges(G, pos, width=[d["weight"]*5 for (_,_,d) in G.edges(data=True)])
            nx.draw_networkx_labels(G, pos, font_size=10)
            plt.title(f"Semantic Similarity Network for '{query}'")
            
            # Save the figure
            output_dir = Path(self.args.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            sim_path = output_dir / f"similarity_network_{query}.png"
            plt.savefig(sim_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Similarity network saved to {sim_path}")
            
            try:
                plt.show()
            except Exception as e:
                self.logger.warning(f"Could not display plot interactively: {e}")

    def create_entity_context_network(self, docs: List[Any], min_occurrences: int = 3, 
                                  output_dir: Optional[str] = None) -> Optional[Any]:
        """
        Creates a force-directed graph visualization showing the relationship between
        medical entities and their contextual attributes.
        
        Args:
            docs: List of processed MedSpaCy documents
            min_occurrences: Minimum number of occurrences to include an entity
            output_dir: Directory to save visualization
            
        Returns:
            NetworkX graph of entity-context relationships or None if networkx not available
        """
        if not HAS_NX:
            self.logger.warning("NetworkX not installed; skipping entity context network.")
            return None
            
        import networkx as nx
        from collections import Counter, defaultdict
        
        self.logger.info("Creating entity context network visualization")
        
        # Track entities and their contexts
        entity_counts = Counter()
        entity_contexts = defaultdict(Counter)
        entity_types = {}
        
        # Process documents
        for doc in tqdm(docs, desc="Analyzing entity contexts"):
            for ent in doc.ents:
                # Count entity occurrences
                entity_text = ent.text.lower()
                entity_counts[entity_text] += 1
                entity_types[entity_text] = ent.label_
                
                # Track context attributes
                if hasattr(ent._, "is_negated") and ent._.is_negated:
                    entity_contexts[entity_text]["negated"] += 1
                if hasattr(ent._, "is_historical") and ent._.is_historical:
                    entity_contexts[entity_text]["historical"] += 1
                if hasattr(ent._, "is_hypothetical") and ent._.is_hypothetical:
                    entity_contexts[entity_text]["hypothetical"] += 1
                if hasattr(ent._, "is_uncertain") and ent._.is_uncertain:
                    entity_contexts[entity_text]["uncertain"] += 1
                if hasattr(ent._, "is_family") and ent._.is_family:
                    entity_contexts[entity_text]["family"] += 1
                
                # Get surrounding context (tokens)
                sent = ent.sent
                if sent:
                    # Look at tokens surrounding the entity
                    start_idx = max(0, ent.start - 5)
                    end_idx = min(len(sent), ent.end + 5)
                    
                    # Extract severity and temporal modifiers
                    for token in sent[start_idx:end_idx]:
                        if token.pos_ == "ADJ" and token.i != ent.root.i:
                            entity_contexts[entity_text][f"modifier:{token.text.lower()}"] += 1
                        if token.pos_ == "ADV" and token.i != ent.root.i:
                            entity_contexts[entity_text][f"modifier:{token.text.lower()}"] += 1
        
        # Filter entities by frequency
        frequent_entities = {entity: count for entity, count in entity_counts.items() 
                        if count >= min_occurrences}
        
        # Create the network
        G = nx.Graph()
        
        # Add entity nodes
        for entity, count in frequent_entities.items():
            G.add_node(entity, 
                    type="entity", 
                    category=entity_types.get(entity, "UNKNOWN"),
                    count=count,
                    size=max(10, min(50, int(count/2))))
        
        # Add context nodes
        context_counts = Counter()
        for entity, contexts in entity_contexts.items():
            if entity in frequent_entities:
                for context, count in contexts.items():
                    if count >= 2:  # Only include contexts that appear at least twice
                        context_counts[context] += count
                        G.add_node(context, type="context", count=count)
                        G.add_edge(entity, context, weight=count)
        
        for context, count in context_counts.items():
            G.nodes[context]['size'] = max(5, min(30, int(count/3)))
        
        self.logger.info(f"Built entity context network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Visualize the network
        if G.number_of_nodes() > 0:
            plt.figure(figsize=(14, 12))
            
            # Choose layout based on graph size
            if G.number_of_nodes() < 100:
                pos = nx.spring_layout(G, seed=42, k=0.3)
            else:
                pos = nx.kamada_kawai_layout(G)
            
            # Define node colors by type
            node_colors = []
            for node in G.nodes():
                if G.nodes[node]['type'] == 'entity':
                    category = G.nodes[node]['category']
                    if category == 'CONDITION':
                        node_colors.append('#ff8c69')  # Orange
                    elif category == 'MEDICATION':
                        node_colors.append('#add8e6')  # Light blue
                    elif category == 'SYMPTOM':
                        node_colors.append('#ffb6c1')  # Pink
                    elif category == 'TREATMENT':
                        node_colors.append('#90ee90')  # Light green
                    else:
                        node_colors.append('#d3d3d3')  # Light gray
                else:
                    # Context nodes
                    if node.startswith('modifier:'):
                        node_colors.append('#f0e68c')  # Khaki
                    elif node == 'negated':
                        node_colors.append('#ff6347')  # Tomato red
                    elif node == 'historical':
                        node_colors.append('#4682b4')  # Steel blue
                    elif node == 'hypothetical':
                        node_colors.append('#9370db')  # Medium purple
                    elif node == 'uncertain':
                        node_colors.append('#ffa500')  # Orange
                    else:
                        node_colors.append('#a9a9a9')  # Dark gray
            
            # Get node sizes from graph
            node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, 
                                node_size=node_sizes, 
                                node_color=node_colors, 
                                alpha=0.8)
            
            # Draw edges with transparency based on weight
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_alphas = [0.1 + 0.8 * (weight / max_weight) for weight in edge_weights]
            
            for (u, v, weight), alpha in zip(G.edges(data='weight'), edge_alphas):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1.5, alpha=alpha)
            
            # Add labels to important nodes
            entity_threshold = max(frequent_entities.values()) * 0.2  # Label top 20% of entities
            entity_labels = {node: node for node in G.nodes() 
                            if G.nodes[node]['type'] == 'entity' and G.nodes[node]['count'] >= entity_threshold}
            
            # Add all context labels
            context_labels = {node: node.replace('modifier:', '') for node in G.nodes() 
                            if G.nodes[node]['type'] == 'context'}
            
            # Combine labels
            labels = {**entity_labels, **context_labels}
            
            # Draw labels with different colors
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, 
                                font_color='black', font_weight='bold')
            
            # Add a legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', label='Condition', 
                    markerfacecolor='#ff8c69', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Medication', 
                    markerfacecolor='#add8e6', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Symptom', 
                    markerfacecolor='#ffb6c1', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Treatment', 
                    markerfacecolor='#90ee90', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Negated', 
                    markerfacecolor='#ff6347', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Historical', 
                    markerfacecolor='#4682b4', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Modifier', 
                    markerfacecolor='#f0e68c', markersize=10)
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title('Entity Context Network: Medical Entities and Their Contextual Attributes')
            plt.axis('off')
            
            # Save visualization
            if output_dir:
                output_path = Path(output_dir) / "entity_context_network.png"
                plt.savefig(output_path, bbox_inches="tight", dpi=300)
                self.logger.info(f"Entity context network saved to {output_path}")
                
                # Save data as JSON for interactive visualization
                try:
                    import json
                    network_data = {
                        "nodes": [{"id": node, 
                                "type": G.nodes[node]['type'],
                                "category": G.nodes[node].get('category', ''),
                                "count": G.nodes[node]['count'],
                                "size": G.nodes[node]['size']} 
                                for node in G.nodes()],
                        "links": [{"source": u, 
                                "target": v, 
                                "value": G[u][v]['weight']} 
                                for u, v in G.edges()]
                    }
                    
                    with open(Path(output_dir) / "entity_context_network.json", 'w') as f:
                        json.dump(network_data, f)
                except Exception as e:
                    self.logger.warning(f"Could not save network data as JSON: {e}")
            
            try:
                plt.show()
            except Exception as e:
                self.logger.warning(f"Could not display plot interactively: {e}")
        else:
            self.logger.warning("No entity contexts met the criteria. No network to visualize.")
        
        return G

def visualize_temporal_evolution(self, notes_df: pd.DataFrame, 
                                    output_dir: Optional[str] = None) -> None:
    """
    Creates a visualization showing how medical concepts evolve over time.
    For patients with multiple notes, it tracks how symptoms, conditions, and treatments 
    change over the course of care.
    
    Args:
        notes_df: DataFrame with processed notes, must include 'chartdate', 'subject_id', 
                'medspacy_entities', and 'medspacy_contexts'
        output_dir: Directory to save visualization
    """
    if not self.args.use_medspacy or 'medspacy_entities' not in notes_df.columns:
        self.logger.warning("MedSpaCy results not available for temporal evolution visualization.")
        return
        
    self.logger.info("Creating temporal evolution visualization of medical concepts")
    
    # Check if 'chartdate' column is available
    if 'chartdate' not in notes_df.columns:
        self.logger.warning("Chart date information not available for temporal visualization.")
        return
    
    # Ensure chartdate is datetime
    try:
        notes_df['chartdate'] = pd.to_datetime(notes_df['chartdate'])
    except Exception as e:
        self.logger.warning(f"Could not convert chartdate to datetime: {e}")
        return
    
    # Group notes by patient (subject_id)
    patient_groups = notes_df.groupby('subject_id')
    
    # Find patients with multiple notes
    patients_with_multiple_notes = []
    for subject_id, group in patient_groups:
        if len(group) >= 3:  # At least 3 notes to show evolution
            patients_with_multiple_notes.append(subject_id)
    
    if not patients_with_multiple_notes:
        self.logger.warning("No patients with multiple notes found for temporal visualization.")
        return
    
    # Select a sample of patients for visualization (max 5)
    sample_patients = patients_with_multiple_notes[:5]
    
    self.logger.info(f"Creating temporal visualizations for {len(sample_patients)} patients")
    
    # Create a directory for patient visualizations
    if output_dir:
        output_path = Path(output_dir) / "temporal_evolution"
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each patient
    for subject_id in sample_patients:
        patient_notes = notes_df[notes_df['subject_id'] == subject_id].sort_values('chartdate')
        
        # Track entities over time
        temporal_data = []
        
        for idx, row in patient_notes.iterrows():
            date = row['chartdate']
            
            # Get entities and their contexts
            note_entities = {}
            
            for entity, label in row['medspacy_entities']:
                entity_text = entity.lower()
                
                # Skip common non-informative entities
                if entity_text in ["patient", "hospital", "normal", "doctor", "noted"]:
                    continue
                
                # Initialize entity data
                if entity_text not in note_entities:
                    note_entities[entity_text] = {
                        'type': label,
                        'is_negated': False,
                        'is_historical': False,
                        'is_hypothetical': False,
                        'count': 0
                    }
                
                note_entities[entity_text]['count'] += 1
            
            # Update with context information if available
            if 'medspacy_contexts' in row:
                for entity, label, is_negated, is_historical, is_hypothetical in row['medspacy_contexts']:
                    entity_text = entity.lower()
                    if entity_text in note_entities:
                        note_entities[entity_text]['is_negated'] = is_negated
                        note_entities[entity_text]['is_historical'] = is_historical
                        note_entities[entity_text]['is_hypothetical'] = is_hypothetical
            
            # Add to temporal data
            for entity, data in note_entities.items():
                temporal_data.append({
                    'date': date,
                    'entity': entity,
                    'type': data['type'],
                    'negated': data['is_negated'],
                    'historical': data['is_historical'],
                    'hypothetical': data['is_hypothetical'],
                    'count': data['count']
                })
        
        # Convert to DataFrame for easier analysis
        if not temporal_data:
            continue
            
        timeline_df = pd.DataFrame(temporal_data)
        
        # Get the most frequently mentioned entities for this patient
        entity_counts = timeline_df.groupby('entity')['count'].sum().reset_index()
        top_entities = entity_counts.sort_values('count', ascending=False).head(10)['entity'].tolist()
        
        # Filter to only include top entities
        timeline_df = timeline_df[timeline_df['entity'].isin(top_entities)]
        
        # Create visualization
        self._create_patient_timeline(timeline_df, subject_id, output_path if output_dir else None)
    
    # Create a summary visualization across all patients
    self._create_summary_timeline(notes_df, sample_patients, output_path if output_dir else None)

def _create_patient_timeline(self, timeline_df: pd.DataFrame, subject_id: int, 
                           output_dir: Optional[Path] = None) -> None:
    """
    Creates a timeline visualization for a single patient.
    
    Args:
        timeline_df: DataFrame with entity mentions over time
        subject_id: Patient ID
        output_dir: Directory to save visualization
    """
    if timeline_df.empty:
        return
        
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Get unique dates and entities
    dates = timeline_df['date'].unique()
    entities = timeline_df['entity'].unique()
    
    # Create a mapping of entity types to colors
    type_colors = {
        'CONDITION': '#DC4343',    # Red
        'DISEASE': '#DC4343',      # Red (same as CONDITION)
        'CHEMICAL': '#FF9561',     # Orange
        'MEDICATION': '#add8e6',   # Light blue
        'TREATMENT': '#90ee90',    # Light green
        'SYMPTOM': '#ffb6c1',      # Pink
        'LAB_RESULT': '#e6e6fa',   # Lavender
        'DIAGNOSTIC_PROCEDURE': '#f0e68c'  # Yellow
    }
    
    # Create plot
    entity_y_positions = {}
    for i, entity in enumerate(entities):
        entity_y_positions[entity] = i
    
    # Draw horizontal lines for each entity
    for entity, y_pos in entity_y_positions.items():
        plt.axhline(y=y_pos, color='gray', linestyle='-', alpha=0.3)
    
    # Plot entity mentions
    for _, row in timeline_df.iterrows():
        entity = row['entity']
        date = row['date']
        entity_type = row['type']
        y_pos = entity_y_positions[entity]
        
        # Determine marker style based on context
        marker = 'o'  # Default
        if row['negated']:
            marker = 'x'  # X for negated
        elif row['historical']:
            marker = 's'  # Square for historical
        elif row['hypothetical']:
            marker = '^'  # Triangle for hypothetical
        
        # Determine color based on entity type
        color = type_colors.get(entity_type, '#d3d3d3')  # Default to light gray
        
        # Plot point
        plt.scatter(date, y_pos, marker=marker, s=100, color=color, alpha=0.7)
    
    # Set y-axis ticks to entity names
    plt.yticks(range(len(entities)), entities)
    
    # Format x-axis as dates
    plt.gcf().autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Medical Concept')
    plt.title(f'Temporal Evolution of Medical Concepts - Patient {subject_id}')
    
    # Add legend
    legend_elements = []
    
    # Entity type legend
    for entity_type, color in type_colors.items():
        if entity_type in timeline_df['type'].values:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=entity_type, 
                                        markerfacecolor=color, markersize=10))
    
    # Context legend
    legend_elements.extend([
        Line2D([0], [0], marker='o', color='w', label='Present', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='x', color='w', label='Negated', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Historical', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Hypothetical', markerfacecolor='gray', markersize=10)
    ])
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_file = output_dir / f"patient_{subject_id}_timeline.png"
        plt.savefig(output_file, bbox_inches="tight", dpi=300)
        self.logger.info(f"Patient timeline saved to {output_file}")
    
    try:
        plt.show()
    except Exception as e:
        self.logger.warning(f"Could not display plot interactively: {e}")
    finally:
        plt.close()

    def _create_summary_timeline(self, notes_df: pd.DataFrame, sample_patients: List, 
                            output_dir: Optional[Path] = None) -> None:
        """
        Creates a summary timeline visualization across multiple patients.
        
        Args:
            notes_df: DataFrame with all notes
            sample_patients: List of patient IDs to include
            output_dir: Directory to save visualization
        """
        if not sample_patients:
            return
            
        # Filter notes to selected patients
        patient_notes = notes_df[notes_df['subject_id'].isin(sample_patients)]
        
        # Create a new dataframe to hold aggregated entity data by date
        entity_data = []
        
        # Group by date and aggregate entity counts
        for date, group in patient_notes.sort_values('chartdate').groupby('chartdate'):
            date_entities = {}
            
            # Process all entities for this date
            for _, row in group.iterrows():
                for entity, label in row['medspacy_entities']:
                    entity_text = entity.lower()
                    entity_type = label
                    
                    # Skip common non-informative entities
                    if entity_text in ["patient", "hospital", "normal", "doctor", "noted"]:
                        continue
                    
                    # Initialize if not exists
                    if entity_text not in date_entities:
                        date_entities[entity_text] = {
                            'type': entity_type,
                            'count': 0,
                            'negated_count': 0
                        }
                    
                    date_entities[entity_text]['count'] += 1
                    
                    # Check if negated
                    if 'medspacy_contexts' in row:
                        for ctx_entity, _, is_negated, _, _ in row['medspacy_contexts']:
                            if ctx_entity.lower() == entity_text and is_negated:
                                date_entities[entity_text]['negated_count'] += 1
            
            # Add to aggregated data
            for entity, data in date_entities.items():
                entity_data.append({
                    'date': date,
                    'entity': entity,
                    'type': data['type'],
                    'count': data['count'],
                    'negated_ratio': data['negated_count'] / data['count'] if data['count'] > 0 else 0
                })
        
        if not entity_data:
            self.logger.warning("No entity data available for summary timeline.")
            return
            
        summary_df = pd.DataFrame(entity_data)
        
        # Get top entities across all patients
        entity_totals = summary_df.groupby('entity')['count'].sum().reset_index()
        top_entities = entity_totals.sort_values('count', ascending=False).head(15)['entity'].tolist()
        
        # Filter to only top entities
        summary_df = summary_df[summary_df['entity'].isin(top_entities)]
        
        # Pivot for heatmap
        pivot_df = summary_df.pivot_table(
            index='entity', 
            columns='date', 
            values='count',
            fill_value=0
        )
        
        # Create heatmap
        plt.figure(figsize=(16, 10))
        
        if HAS_SNS:
            # Create heatmap with seaborn if available
            ax = sns.heatmap(pivot_df, cmap="YlOrRd", linewidths=0.5, linecolor='gray')
            plt.title("Medical Concept Frequency Over Time Across Patients")
            plt.xlabel("Date")
            plt.ylabel("Medical Concept")
            
            # Rotate date labels
            plt.gcf().autofmt_xdate()
        else:
            # Create basic heatmap with matplotlib
            plt.imshow(pivot_df.values, cmap='YlOrRd', aspect='auto')
            plt.colorbar(label='Frequency')
            plt.title("Medical Concept Frequency Over Time Across Patients")
            plt.xlabel("Date")
            plt.ylabel("Medical Concept")
            
            # Set x and y axis ticks
            plt.yticks(range(len(pivot_df.index)), pivot_df.index)
            plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=45)
        
        plt.tight_layout()
        
        # Save if output directory provided
        if output_dir:
            output_file = output_dir / "concept_frequency_heatmap.png"
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            self.logger.info(f"Concept frequency heatmap saved to {output_file}")
        
        try:
            plt.show()
        except Exception as e:
            self.logger.warning(f"Could not display plot interactively: {e}")
        finally:
            plt.close()
        
        # Create line chart for top 5 entities over time
        top5_entities = entity_totals.sort_values('count', ascending=False).head(5)['entity'].tolist()
        top5_df = summary_df[summary_df['entity'].isin(top5_entities)]
        
        plt.figure(figsize=(14, 8))
        
        # Group by date and entity, sum counts
        trend_df = top5_df.groupby(['date', 'entity'])['count'].sum().reset_index()
        
        # Plot each entity as a line
        for entity in top5_entities:
            entity_data = trend_df[trend_df['entity'] == entity]
            plt.plot(entity_data['date'], entity_data['count'], marker='o', linewidth=2, label=entity)
        
        plt.title("Trend of Top 5 Medical Concepts Over Time")
        plt.xlabel("Date")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format x-axis dates
        plt.gcf().autofmt_xdate()
        
        if output_dir:
            output_file = output_dir / "concept_trend_lines.png"
            plt.savefig(output_file, bbox_inches="tight", dpi=300)
            self.logger.info(f"Concept trend lines saved to {output_file}")
        
        try:
            plt.show()
        except Exception as e:
            self.logger.warning(f"Could not display plot interactively: {e}")

# ------------------- Main Execution ------------------- #
def main():
    start_time = datetime.now()
    try:
        args = parse_arguments()
        pipeline = MIMICNLPPipeline(args)
        pipeline.logger.info(f"Starting enhanced MIMIC NLP pipeline for ICD-9 codes {args.icd9_codes}")
        
        # Step 1: Load data
        pipeline.logger.info("Step 1: Loading data from MIMIC database")
        notes_df = pipeline.load_data()
        if len(notes_df) > 10000:
            pipeline.logger.warning("Large dataset detected. Ensure sufficient memory is available.")
        pipeline.logger.info(f"Loaded {len(notes_df)} notes from the database")
        
        # Step 2: Process documents with SpaCy, SciSpaCy, and optionally MedSpaCy
        pipeline.logger.info("Step 2: Processing documents with NLP models")
        results = pipeline.process_documents(texts=notes_df['text'].tolist(), batch_size=args.batch_size)
        notes_df['clean_text'] = results['clean_texts']
        notes_df['spacy_entities'] = results['spacy_entities']
        notes_df['scispacy_entities'] = results['scispacy_entities']
        
        # Save MedSpaCy results if enabled
        if args.use_medspacy and 'medspacy_entities' in results:
            notes_df['medspacy_entities'] = results['medspacy_entities']
            notes_df['medspacy_contexts'] = results['medspacy_contexts']
            notes_df['medspacy_sections'] = results['medspacy_sections']
            
            # Log MedSpaCy results
            total_medspacy_entities = sum(len(ents) for ents in results['medspacy_entities'])
            pipeline.logger.info(f"Extracted {total_medspacy_entities} entities with MedSpaCy")
        
        # Save processed documents for visualization
        spacy_docs = results['spacy_docs']
        scispacy_docs = results['scispacy_docs']
        
        # Store MedSpaCy docs if available
        medspacy_docs = results.get('medspacy_docs', [])
        
        # Log entity extraction results
        total_spacy_entities = sum(len(ents) for ents in results['spacy_entities'])
        total_scispacy_entities = sum(len(ents) for ents in results['scispacy_entities'])
        pipeline.logger.info(f"Extracted {total_spacy_entities} entities with SpaCy")
        pipeline.logger.info(f"Extracted {total_scispacy_entities} entities with SciSpacy")
        
        # Clean up memory
        del results
        gc.collect()
        if torch.cuda.is_available() and not args.disable_gpu:
            torch.cuda.empty_cache()
        
        # Step 3: Entity overlap analysis
        pipeline.logger.info("Step 3: Analyzing entity overlap between models")
        overlap_df = pipeline.analyze_entity_overlap(notes_df)
        
        # Report overlap statistics
        total_overlap = overlap_df['overlap_count'].sum()
        total_entities = total_spacy_entities + total_scispacy_entities
        
        if args.use_medspacy and 'medspacy_count' in overlap_df.columns:
            total_medspacy_entities = overlap_df['medspacy_count'].sum()
            total_entities += total_medspacy_entities
            total_entities -= overlap_df['overlap_all_models'].sum()
            avg_overlap_pct = 100 * overlap_df['overlap_all_models'].sum() / total_entities
            pipeline.logger.info(f"Average overlap across all models: {avg_overlap_pct:.2f}%")
        else:
            avg_overlap_pct = 100 * total_overlap / (total_entities - total_overlap)
            pipeline.logger.info(f"Average entity overlap between SpaCy and SciSpacy: {avg_overlap_pct:.2f}%")
        
        # Step 4: MedSpaCy analysis if enabled
        if args.use_medspacy and medspacy_docs:
            pipeline.logger.info("Step 4: Analyzing MedSpaCy results")
            
            # Visualize MedSpaCy sections
            pipeline.visualize_medspacy_sections(
                docs=medspacy_docs,
                num_examples=args.visualize_examples,
                output_dir=args.output_path
            )
            
            # Visualize MedSpaCy context
            pipeline.visualize_medspacy_context(
                docs=medspacy_docs,
                num_examples=args.visualize_examples,
                output_dir=args.output_path
            )
            
            # Analyze MedSpaCy findings
            medspacy_analysis = pipeline.analyze_medspacy_findings(
                docs=medspacy_docs,
                output_dir=args.output_path
            )
            
            # Compare entity recognition between models
            pipeline.compare_entity_recognition(
                notes_df=notes_df,
                output_dir=args.output_path
            )
            
            pipeline.logger.info("MedSpaCy analysis completed")
        
        # Step 5: Enhanced entity processing if requested
        if args.visualize_entities or args.extract_relationships:
            pipeline.logger.info("Step 5: Performing enhanced entity processing")
            enhanced_entities = pipeline.enhance_entity_extraction(
                texts=notes_df['text'].tolist(), 
                batch_size=args.batch_size
            )
            
            # Analyze entity frequencies
            entity_frequencies = pipeline.analyze_entity_frequencies(enhanced_entities)
            pipeline.logger.info("Entity frequency analysis completed")
            
            # Visualize entities with displaCy if requested
            if args.visualize_entities:
                pipeline.logger.info("Creating entity visualizations with displaCy")
                pipeline.visualize_entities_with_displacy(
                    docs=scispacy_docs, 
                    num_examples=args.visualize_examples,
                    output_dir=args.output_path
                )
                
                # Create entity summary visualization
                pipeline.create_entity_summary_visualization(
                    entities_by_type=entity_frequencies, 
                    output_dir=args.output_path
                )
                
                # Create entity co-occurrence network
                if HAS_NX:
                    pipeline.logger.info("Creating entity co-occurrence network")
                    pipeline.create_entity_co_occurrence_network(
                        entity_data=enhanced_entities,
                        min_co_occurrences=3,
                        output_dir=args.output_path
                    )
            
            # Extract relationships using dependency parsing
            if args.extract_relationships:
                pipeline.logger.info("Step 6: Extracting entity relationships using dependency parsing")
                relationships = pipeline.extract_entity_relationships(
                    texts=notes_df['text'].tolist(), 
                    batch_size=args.batch_size
                )
                
                pipeline.logger.info(f"Extracted {len(relationships)} relationships between entities")
                
                # Visualize dependency trees
                pipeline.visualize_dependency_relations(
                    docs=scispacy_docs,
                    num_examples=args.visualize_examples,
                    output_dir=args.output_path
                )
                
                # Build knowledge graph
                if HAS_NX and relationships:
                    pipeline.logger.info("Building knowledge graph from entity relationships")
                    pipeline.build_relationship_knowledge_graph(
                        relationships=relationships, 
                        output_dir=args.output_path
                    )
        
        # Step 7: Word embeddings
        pipeline.logger.info("Step 7: Handling word embeddings")
        # Depending on embedding_choice, train custom model, load pretrained model, or both.
        custom_model = None
        pretrained_model = None
        if args.embedding_choice in ["custom", "both"]:
            custom_model = pipeline.train_word2vec(scispacy_entities=notes_df['scispacy_entities'])
        if args.embedding_choice in ["pretrained", "both"]:
            pretrained_model = pipeline.load_pretrained_embeddings(args.pretrained_path)
        
        # Compare embeddings if both models are available
        if custom_model is not None and pretrained_model is not None:
            # Try with different query terms
            for query in ["sepsis", "infection", "bacteria", "patient"]:
                try:
                    if (query in custom_model.wv.key_to_index and 
                        query in pretrained_model.key_to_index):
                        pipeline.compare_embeddings(custom_model, pretrained_model, query=query, topn=10)
                        break
                except Exception:
                    continue
        
        # Visualize embeddings: use custom model if available; otherwise, use pretrained.
        emb_model = custom_model if custom_model is not None else pretrained_model
        if emb_model is not None and not args.no_tsne:
            pipeline.visualize_embeddings_multiple(emb_model, topn=200)
        
        # Step 8: Semantic similarity analysis
        pipeline.logger.info("Step 8: Performing semantic similarity analysis")
        if emb_model is not None:
            # Try with different query terms
            for query in ["sepsis", "infection", "bacteria", "antibiotic", "patient"]:
                try:
                    if isinstance(emb_model, Word2Vec):
                        if query in emb_model.wv.key_to_index:
                            pipeline.semantic_similarity_analysis(emb_model, query=query, topn=10)
                            break
                    else:
                        if query in emb_model.key_to_index:
                            pipeline.semantic_similarity_analysis(emb_model, query=query, topn=10)
                            break
                except Exception:
                    continue
        
        # Completion timing
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        hours, remainder = divmod(execution_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        pipeline.logger.info(f"Pipeline completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        pipeline.logger.info(f"Results saved to {args.output_path}")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logging.info("GPU cache cleared")
            except Exception:
                pass

if __name__ == "__main__":
    main()

# Instructions for running the script with MedSpaCy
"""
-- Run the script with the --use_medspacy flag:
   python mimic_nlp.py --db_path "path/to/mimic.duckdb" --use_medspacy --output_path results

-- For the full analysis with all options:
   python mimic_nlp.py --db_path "path/to/mimic.duckdb" --icd9_codes "995.91,038,486" --note_category "Nursing/other" --embedding_choice both --visualize_entities --extract_relationships --output_path results --use_medspacy

The results will include MedSpaCy-specific visualizations and analyses:
- Section detection in clinical notes
- Context attributes (negation, historical, hypothetical status)
- Comparison with SpaCy and SciSpaCy entity extraction
- Integrated entity visualizations
"""