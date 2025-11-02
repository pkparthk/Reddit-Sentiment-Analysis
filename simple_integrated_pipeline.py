import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

PYSPARK_AVAILABLE = False
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import col, mean, count, desc
    PYSPARK_AVAILABLE = True
    print("✅ PySpark available - will attempt distributed processing")
except ImportError:
    print("⚠️  PySpark not available - using pandas for processing")

# ML imports for sentiment analysis
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    ML_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers not available - will use simple sentiment")
    ML_AVAILABLE = False

# Configure logging without emojis for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SimplifiedIntegratedPipeline:
    """
    Simplified pipeline that integrates fetch.js with data processing
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.pipeline_id = f"pipeline_{int(time.time())}"
        self.spark = None
        self.sentiment_model = None
        self.tokenizer = None
        self.analyzer_instance = None
        self.use_spark = False
        
        logger.info(f"Initializing Pipeline: {self.pipeline_id}")
        logger.info(f"PySpark Available: {PYSPARK_AVAILABLE}")
        logger.info(f"ML Available: {ML_AVAILABLE}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            "reddit": {
                "batch_mode": True,
                "timeout": 60
            },
            "processing": {
                "use_spark": PYSPARK_AVAILABLE,
                "partitions": 4
            },
            "sentiment": {
                "model_name": "barissayil/bert-sentiment-analysis-sst",
                "batch_size": 8,
                "use_simple_fallback": True
            },
            "output": {
                "save_detailed": True,
                "save_summary": True
            }
        }
    
    def initialize_spark(self) -> bool:
        """Try to initialize Spark, fall back gracefully"""
        if not PYSPARK_AVAILABLE:
            logger.info("Spark not available - using pandas processing")
            return False
            
        try:
            logger.info("Attempting to initialize Spark...")
            self.spark = (SparkSession.builder
                         .appName("RedditSentiment")
                         .master("local[2]")
                         .config("spark.sql.adaptive.enabled", "true")
                         .config("spark.driver.host", "localhost")
                         .getOrCreate())
            
            self.spark.sparkContext.setLogLevel("ERROR")
            self.use_spark = True
            logger.info("Spark initialized successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Spark initialization failed: {str(e)[:100]}...")
            logger.info("Falling back to pandas processing")
            return False
    
    def initialize_sentiment_model(self) -> bool:
        """Initialize sentiment model or use simple fallback"""
        if not ML_AVAILABLE:
            logger.info("Using simple sentiment fallback (transformers not available)")
            return True
        
        # Try to use existing analyzer.py model first
        try:
            logger.info("Attempting to use existing analyzer.py model...")
            sys.path.append(os.getcwd())
            from analyzer import Analyzer
            
            # Create a simple args object with all required attributes
            class SimpleArgs:
                def __init__(self):
                    self.model_name_or_path = "barissayil/bert-sentiment-analysis-sst"
                    self.maxlen_val = 512
                    self.output_dir = "sentiment_model"
                    self.batch_size = 32
                    self.maxlen_train = 30
            
            # Initialize the analyzer from existing code
            args = SimpleArgs()
            self.analyzer_instance = Analyzer(will_train=False, args=args)
            logger.info("Successfully loaded existing analyzer model")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load existing analyzer: {e}")
        
        # Fallback to direct BERT loading
        try:
            logger.info("Loading BERT sentiment model directly...")
            model_name = self.config["sentiment"]["model_name"]
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.sentiment_model.eval()
            
            logger.info("BERT sentiment model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")
            logger.info("Using simple sentiment fallback")
            return True
    
    def fetch_reddit_data(self) -> bool:
        """Execute fetch.js to get Reddit comments or use existing data"""
        # Check if we already have data
        existing_files = ["comments.json", "reddit_comments_combined.jsonl", "reddit_comments_spark.jsonl"]
        for file in existing_files:
            if os.path.exists(file):
                logger.info(f"Using existing Reddit data: {file}")
                return True
        
        try:
            logger.info("Fetching fresh Reddit comments using fetch.js...")
            
            cmd = ["node", "fetch.js"]
            if self.config["reddit"]["batch_mode"]:
                cmd.append("--batch")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config["reddit"]["timeout"]
            )
            
            if result.returncode != 0:
                logger.error(f"fetch.js failed: {result.stderr}")
                return False
            
            logger.info("Reddit data fetching completed")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("fetch.js timed out")
            return False
        except Exception as e:
            logger.error(f"Error executing fetch.js: {e}")
            return False
    
    def load_reddit_data(self) -> Optional[pd.DataFrame]:
        """Load Reddit data into DataFrame"""
        try:
            logger.info("Loading Reddit data...")
            
            # Check for available data files in order of preference
            data_files = []
            for file in ["reddit_comments_combined.jsonl", "reddit_comments_spark.jsonl", "comments.json"]:
                if os.path.exists(file):
                    data_files.append(file)
            
            if not data_files:
                logger.error("No Reddit data files found")
                return None
            
            data_file = data_files[0]
            logger.info(f"Loading: {data_file}")
            
            comments = []
            
            # Handle different file formats
            if data_file.endswith('.jsonl'):
                # JSONL format
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                comments.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue
            else:
                # JSON format (like comments.json)
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                comment_data = json.loads(line.strip())
                                # Convert to standard format
                                standardized = {
                                    "id": f"comment_{len(comments)}",
                                    "author": "unknown",
                                    "text": comment_data.get("text", ""),
                                    "score": 1,
                                    "created_utc": time.time(),
                                    "subreddit": "IndianWorkplace",
                                    "post_id": "extracted",
                                    "extraction_timestamp": datetime.now().isoformat(),
                                    "comment_length": len(comment_data.get("text", "")),
                                    "word_count": len(comment_data.get("text", "").split()),
                                    "contains_url": "http" in comment_data.get("text", ""),
                                    "is_question": "?" in comment_data.get("text", ""),
                                    "sentiment_label": None,
                                    "confidence_score": None
                                }
                                comments.append(standardized)
                            except json.JSONDecodeError:
                                continue
            
            df = pd.DataFrame(comments)
            logger.info(f"Loaded {len(df)} comments")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess Reddit data"""
        try:
            logger.info("Preprocessing data...")
            
            # Filter valid comments
            df_clean = df[
                (df['text'].notna()) & 
                (df['text'] != '') & 
                (df['text'] != '[deleted]') &
                (df['text'] != '[removed]')
            ].copy()
            
            # Add features
            df_clean['text_length_category'] = pd.cut(
                df_clean['comment_length'], 
                bins=[0, 50, 200, float('inf')], 
                labels=['short', 'medium', 'long']
            )
            
            df_clean['engagement_score'] = df_clean['score'] + df_clean['word_count'] * 0.1
            
            logger.info(f"Preprocessed {len(df_clean)} comments")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return df
    
    def simple_sentiment_analysis(self, text: str) -> Tuple[str, float]:
        """Enhanced rule-based sentiment analysis fallback"""
        positive_words = ['good', 'great', 'excellent', 'awesome', 'love', 'like', 'happy', 'best', 'amazing', 'wonderful', 'fantastic', 'perfect', 'brilliant', 'outstanding', 'superb', 'appreciate', 'thanks', 'respect', 'congratulations']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'worst', 'horrible', 'disappointing', 'frustrating', 'annoying', 'stupid', 'useless', 'fail', 'failed', 'wrong', 'problem', 'difficult', 'hard', 'struggle']
        
        # Sentiment indicators
        question_words = ['?', 'how', 'what', 'why', 'when', 'where']
        negative_indicators = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 'neither', 'cannot', "can't", "don't", "won't", "shouldn't", "couldn't"]
        
        text_lower = text.lower()
        
        # Count sentiment words
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        question_count = sum(1 for word in question_words if word in text_lower)
        negative_indicator_count = sum(1 for word in negative_indicators if word in text_lower)
        
        # Calculate sentiment score
        sentiment_score = pos_count - neg_count - (negative_indicator_count * 0.5)
        
        # Determine sentiment
        if sentiment_score > 0.5:
            confidence = min(65 + sentiment_score * 8, 92)
            return "Positive", confidence
        elif sentiment_score < -0.5:
            confidence = min(65 + abs(sentiment_score) * 8, 92)
            return "Negative", confidence
        else:
            # Neutral with slight variation based on text characteristics
            if question_count > 0:
                return "Neutral", 55.0  # Questions tend to be neutral
            elif len(text.split()) > 50:
                return "Negative", 58.0  # Long texts might indicate complaints
            else:
                return "Positive", 62.0  # Short texts tend to be more positive
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Analyze sentiment for batch of texts"""
        # Try existing analyzer first
        if hasattr(self, 'analyzer_instance') and self.analyzer_instance is not None:
            try:
                results = []
                for text in texts:
                    # Use the correct method name from analyzer.py
                    label, confidence = self.analyzer_instance.classify_sentiment(text)
                    results.append((label, float(confidence)))
                return results
            except Exception as e:
                logger.warning(f"Analyzer instance failed: {e}, trying BERT")
        
        # Try BERT model
        if ML_AVAILABLE and self.sentiment_model is not None:
            try:
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                
                with torch.no_grad():
                    outputs = self.sentiment_model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                results = []
                for pred in predictions:
                    confidence = float(torch.max(pred)) * 100
                    label = "Positive" if pred[1] > pred[0] else "Negative"
                    results.append((label, confidence))
                
                return results
                
            except Exception as e:
                logger.warning(f"BERT analysis failed: {e}, using simple fallback")
        
        # Fallback to simple analysis
        return [self.simple_sentiment_analysis(text) for text in texts]
    
    def perform_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform sentiment analysis on DataFrame"""
        try:
            logger.info("Performing sentiment analysis...")
            
            texts = df['text'].tolist()
            batch_size = self.config.get("sentiment", {}).get("batch_size", 8)
            
            all_results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                try:
                    batch_results = self.analyze_sentiment_batch(batch_texts)
                    all_results.extend(batch_results)
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
                except Exception as batch_error:
                    logger.warning(f"Batch processing error: {batch_error}")
                    # Use fallback for this batch
                    fallback_results = [self.simple_sentiment_analysis(text) for text in batch_texts]
                    all_results.extend(fallback_results)
            
            # Ensure we have results for all texts
            if len(all_results) != len(texts):
                logger.warning(f"Results mismatch: {len(all_results)} results for {len(texts)} texts")
                # Pad with neutral sentiments if needed
                while len(all_results) < len(texts):
                    all_results.append(("Neutral", 50.0))
            
            # Add results to DataFrame
            df_result = df.copy()
            df_result['sentiment_label'] = [r[0] for r in all_results]
            df_result['confidence_score'] = pd.Series([r[1] for r in all_results], dtype='float64')
            
            logger.info(f"Sentiment analysis completed: {len(all_results)} results")
            return df_result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return original DataFrame with default sentiment values
            df_copy = df.copy()
            df_copy['sentiment_label'] = 'Neutral'
            df_copy['confidence_score'] = 50.0
            return df_copy
    
    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform analytics on results"""
        try:
            logger.info("Analyzing results...")
            
            # Ensure confidence_score is numeric
            df['confidence_score'] = pd.to_numeric(df['confidence_score'], errors='coerce').fillna(50.0)
            
            analytics = {
                "pipeline_metadata": {
                    "pipeline_id": self.pipeline_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_comments": len(df),
                    "processing_method": "Spark" if self.use_spark else "Pandas"
                },
                "sentiment_distribution": df['sentiment_label'].value_counts().to_dict(),
                "average_confidence": df.groupby('sentiment_label')['confidence_score'].mean().to_dict(),
                "text_statistics": {
                    "avg_length": float(df['comment_length'].mean()) if 'comment_length' in df.columns else 0.0,
                    "avg_words": float(df['word_count'].mean()) if 'word_count' in df.columns else 0.0,
                    "avg_score": float(df['score'].mean()) if 'score' in df.columns else 0.0
                }
            }
            
            # Get top examples safely
            try:
                pos_df = df[df['sentiment_label'] == 'Positive']
                neg_df = df[df['sentiment_label'] == 'Negative']
                
                analytics["top_examples"] = {
                    "positive": pos_df.nlargest(min(3, len(pos_df)), 'confidence_score')[['text', 'confidence_score']].to_dict('records') if len(pos_df) > 0 else [],
                    "negative": neg_df.nlargest(min(3, len(neg_df)), 'confidence_score')[['text', 'confidence_score']].to_dict('records') if len(neg_df) > 0 else []
                }
                
                # Truncate text in examples
                for category in ['positive', 'negative']:
                    for example in analytics['top_examples'][category]:
                        example['text'] = example['text'][:100] + "..." if len(example['text']) > 100 else example['text']
            
            except Exception as ex:
                logger.warning(f"Could not generate top examples: {ex}")
                analytics["top_examples"] = {"positive": [], "negative": []}
            
            logger.info("Analytics completed")
            return analytics
            
        except Exception as e:
            logger.error(f"Error in analytics: {e}")
            return {
                "pipeline_metadata": {
                    "pipeline_id": self.pipeline_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_comments": len(df),
                    "processing_method": "Pandas (fallback)"
                }
            }
    
    def save_results(self, df: pd.DataFrame, analytics: Dict) -> bool:
        """Save results to files"""
        try:
            logger.info("Saving results...")
            
            # Create output directory if it doesn't exist
            os.makedirs("output", exist_ok=True)
            
            # Save detailed results
            if self.config.get("output", {}).get("save_detailed", True):
                df.to_json("output/detailed_results.json", orient="records", indent=2)
                df.to_csv("output/results.csv", index=False)
                logger.info("Detailed results saved to output/")
            
            # Save analytics
            if self.config.get("output", {}).get("save_summary", True):
                with open("output/analytics_summary.json", "w") as f:
                    json.dump(analytics, f, indent=2, default=str)
                logger.info("Analytics summary saved to output/")
            
            logger.info("Results saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Execute the complete pipeline"""
        try:
            start_time = time.time()
            logger.info("Starting Integrated Reddit Sentiment Analysis Pipeline")
            logger.info("=" * 60)
            
            # Step 1: Initialize components
            self.initialize_spark()  # May fail, that's OK
            self.initialize_sentiment_model()
            
            # Step 2: Fetch Reddit data
            if not self.fetch_reddit_data():
                logger.error("Failed to fetch Reddit data")
                return False
            
            # Step 3: Load data
            df = self.load_reddit_data()
            if df is None:
                return False
            
            # Step 4: Preprocess
            df_processed = self.preprocess_data(df)
            
            # Step 5: Sentiment analysis
            df_with_sentiment = self.perform_sentiment_analysis(df_processed)
            
            # Step 6: Analytics
            analytics = self.analyze_results(df_with_sentiment)
            
            # Step 7: Save results
            self.save_results(df_with_sentiment, analytics)
            
            # Step 8: Display summary
            execution_time = time.time() - start_time
            self._display_summary(analytics, execution_time)
            
            logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False
        finally:
            if self.spark:
                self.spark.stop()
    
    def _display_summary(self, analytics: Dict, execution_time: float):
        """Display pipeline summary"""
        print("\n" + "=" * 60)
        print("INTEGRATED PIPELINE RESULTS")
        print("=" * 60)
        
        metadata = analytics.get("pipeline_metadata", {})
        print(f"Pipeline ID: {metadata.get('pipeline_id', 'N/A')}")
        print(f"Total Comments: {metadata.get('total_comments', 0)}")
        print(f"Processing Method: {metadata.get('processing_method', 'Unknown')}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        sentiment_dist = analytics.get("sentiment_distribution", {})
        if sentiment_dist:
            print(f"\nSENTIMENT DISTRIBUTION:")
            for sentiment, count in sentiment_dist.items():
                percentage = (count / metadata.get('total_comments', 1)) * 100
                print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        confidence = analytics.get("average_confidence", {})
        if confidence:
            print(f"\nAVERAGE CONFIDENCE:")
            for sentiment, conf in confidence.items():
                print(f"  {sentiment}: {conf:.1f}%")
        
        print("=" * 60)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Simplified Integrated Reddit Sentiment Pipeline")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing")
    parser.add_argument("--no-spark", action="store_true", help="Disable Spark even if available")
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        "reddit": {"batch_mode": args.batch},
        "processing": {"use_spark": PYSPARK_AVAILABLE and not args.no_spark}
    }
    
    # Run pipeline
    pipeline = SimplifiedIntegratedPipeline(config)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()