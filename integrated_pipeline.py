"""
Integrated PySpark Reddit Sentiment Analysis Pipeline
Comprehensive big data pipeline that integrates fetch.js with PySpark processing
"""

import os
import sys
import json
import time
import subprocess
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

# PySpark imports
try:
    from pyspark.sql import SparkSession, DataFrame
    from pyspark.sql.functions import (
        col, udf, length, split, size, regexp_extract, 
        when, isnan, isnull, count, mean, stddev,
        desc, asc, collect_list
    )
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, 
        DoubleType, TimestampType, BooleanType
    )
    from pyspark.ml.feature import Tokenizer, StopWordsRemover
    PYSPARK_AVAILABLE = True
except ImportError:
    print("⚠️  PySpark not available. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pyspark"])
    PYSPARK_AVAILABLE = False

# ML imports for sentiment analysis
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pyspark_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class IntegratedRedditSentimentPipeline:
    """
    Complete integrated pipeline: fetch.js -> PySpark processing -> sentiment analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.pipeline_id = f"integrated_pipeline_{int(time.time())}"
        self.spark = None
        self.sentiment_model = None
        self.tokenizer = None
        self.results = {}
        
        # Initialize logging
        logger.info(f"🚀 Initializing Integrated Pipeline: {self.pipeline_id}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            "spark": {
                "app_name": "RedditSentimentAnalysis",
                "master": "local[*]",
                "memory": "4g",
                "cores": 4
            },
            "reddit": {
                "batch_mode": True,
                "max_urls": 5,
                "delay_ms": 1000
            },
            "sentiment": {
                "model_name": "barissayil/bert-sentiment-analysis-sst",
                "batch_size": 16,
                "max_length": 512
            },
            "output": {
                "save_intermediate": True,
                "output_format": "json",
                "include_metadata": True
            }
        }
    
    def initialize_spark(self) -> bool:
        """Initialize Spark session with optimized configuration"""
        try:
            logger.info("🔧 Initializing Spark session...")
            
            spark_config = self.config["spark"]
            self.spark = (SparkSession.builder
                         .appName(spark_config["app_name"])
                         .master(spark_config["master"])
                         .config("spark.executor.memory", spark_config["memory"])
                         .config("spark.sql.adaptive.enabled", "true")
                         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                         .getOrCreate())
            
            # Set log level to reduce verbosity
            self.spark.sparkContext.setLogLevel("WARN")
            
            logger.info(f"✅ Spark session initialized: {self.spark.version}")
            logger.info(f"   Master: {spark_config['master']}")
            logger.info(f"   Memory: {spark_config['memory']}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Spark: {e}")
            return False
    
    def initialize_sentiment_model(self) -> bool:
        """Initialize BERT sentiment analysis model"""
        try:
            logger.info("🤖 Loading BERT sentiment model...")
            
            model_name = self.config["sentiment"]["model_name"]
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.sentiment_model.eval()
            
            logger.info(f"✅ Sentiment model loaded: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            return False
    
    def fetch_reddit_data(self) -> bool:
        """Execute fetch.js to get Reddit comments"""
        try:
            logger.info("📡 Fetching Reddit comments using fetch.js...")
            
            # Determine batch mode
            batch_flag = "--batch" if self.config["reddit"]["batch_mode"] else ""
            
            # Execute fetch.js
            cmd = ["node", "fetch.js"]
            if batch_flag:
                cmd.append(batch_flag)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.getcwd(),
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode != 0:
                logger.error(f"❌ fetch.js failed: {result.stderr}")
                return False
            
            logger.info("✅ Reddit data fetching completed")
            logger.info(f"   Output: {result.stdout.split('✅')[0].strip() if '✅' in result.stdout else result.stdout[:200]}...")
            
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("❌ fetch.js timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error executing fetch.js: {e}")
            return False
    
    def load_reddit_data_to_spark(self) -> Optional[DataFrame]:
        """Load Reddit data into Spark DataFrame"""
        try:
            logger.info("📊 Loading Reddit data into Spark DataFrame...")
            
            # Check for available data files
            data_files = []
            for file in ["reddit_comments_combined.jsonl", "reddit_comments_spark.jsonl"]:
                if os.path.exists(file):
                    data_files.append(file)
            
            if not data_files:
                logger.error("❌ No Reddit data files found")
                return None
            
            # Load the first available file
            data_file = data_files[0]
            logger.info(f"   Loading: {data_file}")
            
            # Define schema for better performance
            schema = StructType([
                StructField("id", StringType(), True),
                StructField("author", StringType(), True),
                StructField("text", StringType(), True),
                StructField("score", IntegerType(), True),
                StructField("created_utc", DoubleType(), True),
                StructField("subreddit", StringType(), True),
                StructField("post_id", StringType(), True),
                StructField("extraction_timestamp", StringType(), True),
                StructField("comment_length", IntegerType(), True),
                StructField("word_count", IntegerType(), True),
                StructField("contains_url", BooleanType(), True),
                StructField("is_question", BooleanType(), True),
                StructField("sentiment_label", StringType(), True),
                StructField("confidence_score", DoubleType(), True)
            ])
            
            # Load JSONL file
            df = self.spark.read.schema(schema).json(data_file)
            
            # Data quality checks
            total_rows = df.count()
            non_empty_text = df.filter(col("text").isNotNull() & (col("text") != "")).count()
            
            logger.info(f"✅ Loaded {total_rows} comments")
            logger.info(f"   Valid text comments: {non_empty_text}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data to Spark: {e}")
            return None
    
    def preprocess_data(self, df: DataFrame) -> DataFrame:
        """Preprocess Reddit data using Spark"""
        try:
            logger.info("🔧 Preprocessing data with Spark...")
            
            # Filter out invalid comments
            df_clean = df.filter(
                col("text").isNotNull() & 
                (col("text") != "") & 
                (col("text") != "[deleted]") &
                (col("text") != "[removed]")
            )
            
            # Add additional features
            df_enhanced = df_clean.withColumn("text_length_category",
                when(col("comment_length") < 50, "short")
                .when(col("comment_length") < 200, "medium")
                .otherwise("long")
            ).withColumn("engagement_score",
                col("score") + col("word_count") * 0.1
            )
            
            # Cache for performance
            df_enhanced.cache()
            
            processed_count = df_enhanced.count()
            logger.info(f"✅ Preprocessed {processed_count} comments")
            
            return df_enhanced
            
        except Exception as e:
            logger.error(f"❌ Error preprocessing data: {e}")
            return df
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Analyze sentiment for a batch of texts"""
        try:
            if not texts:
                return []
            
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config["sentiment"]["max_length"],
                return_tensors="pt"
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            results = []
            for i, pred in enumerate(predictions):
                confidence = float(torch.max(pred))
                label = "Positive" if pred[1] > pred[0] else "Negative"
                results.append((label, confidence * 100))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in sentiment analysis: {e}")
            return [("Unknown", 0.0)] * len(texts)
    
    def perform_sentiment_analysis(self, df: DataFrame) -> DataFrame:
        """Perform distributed sentiment analysis on Spark DataFrame"""
        try:
            logger.info("🎭 Performing sentiment analysis...")
            
            # Collect texts for sentiment analysis
            texts_df = df.select("id", "text").collect()
            
            # Process in batches
            batch_size = self.config["sentiment"]["batch_size"]
            results = []
            
            for i in range(0, len(texts_df), batch_size):
                batch = texts_df[i:i + batch_size]
                batch_texts = [row.text for row in batch]
                batch_ids = [row.id for row in batch]
                
                logger.info(f"   Processing batch {i//batch_size + 1}/{(len(texts_df) + batch_size - 1)//batch_size}")
                
                sentiment_results = self.analyze_sentiment_batch(batch_texts)
                
                for j, (label, confidence) in enumerate(sentiment_results):
                    results.append({
                        "id": batch_ids[j],
                        "sentiment_label": label,
                        "confidence_score": confidence
                    })
            
            # Create DataFrame with results
            sentiment_df = self.spark.createDataFrame(results)
            
            # Join with original DataFrame
            final_df = df.join(sentiment_df, on="id", how="left")
            
            logger.info("✅ Sentiment analysis completed")
            
            return final_df
            
        except Exception as e:
            logger.error(f"❌ Error in sentiment analysis: {e}")
            return df
    
    def analyze_results(self, df: DataFrame) -> Dict[str, Any]:
        """Perform comprehensive analytics on results"""
        try:
            logger.info("📈 Analyzing results...")
            
            # Basic statistics
            total_comments = df.count()
            
            # Sentiment distribution
            sentiment_dist = df.groupBy("sentiment_label").count().collect()
            sentiment_dict = {row.sentiment_label: row['count'] for row in sentiment_dist}
            
            # Average confidence by sentiment
            avg_confidence = df.groupBy("sentiment_label").agg(mean("confidence_score")).collect()
            confidence_dict = {row.sentiment_label: float(row['avg(confidence_score)']) for row in avg_confidence}
            
            # Text length analysis
            length_stats = df.select(
                mean("comment_length").alias("avg_length"),
                stddev("comment_length").alias("std_length")
            ).collect()[0]
            
            # Engagement analysis
            engagement_stats = df.select(
                mean("score").alias("avg_score"),
                mean("word_count").alias("avg_words")
            ).collect()[0]
            
            # Top engaging comments by sentiment
            top_positive = df.filter(col("sentiment_label") == "Positive").orderBy(desc("confidence_score")).limit(3).collect()
            top_negative = df.filter(col("sentiment_label") == "Negative").orderBy(desc("confidence_score")).limit(3).collect()
            
            analytics = {
                "pipeline_metadata": {
                    "pipeline_id": self.pipeline_id,
                    "timestamp": datetime.now().isoformat(),
                    "total_comments": total_comments
                },
                "sentiment_distribution": sentiment_dict,
                "average_confidence": confidence_dict,
                "text_statistics": {
                    "avg_length": float(length_stats.avg_length) if length_stats.avg_length else 0,
                    "std_length": float(length_stats.std_length) if length_stats.std_length else 0,
                    "avg_words": float(engagement_stats.avg_words) if engagement_stats.avg_words else 0,
                    "avg_score": float(engagement_stats.avg_score) if engagement_stats.avg_score else 0
                },
                "top_examples": {
                    "positive": [{"text": row.text[:100], "confidence": row.confidence_score} for row in top_positive],
                    "negative": [{"text": row.text[:100], "confidence": row.confidence_score} for row in top_negative]
                }
            }
            
            logger.info("✅ Analytics completed")
            return analytics
            
        except Exception as e:
            logger.error(f"❌ Error in analytics: {e}")
            return {}
    
    def save_results(self, df: DataFrame, analytics: Dict) -> bool:
        """Save results to various formats"""
        try:
            logger.info("💾 Saving results...")
            
            # Save Spark DataFrame as JSON
            df.coalesce(1).write.mode("overwrite").json("output/spark_results")
            
            # Save analytics
            with open("pyspark_analytics.json", "w") as f:
                json.dump(analytics, f, indent=2)
            
            # Save detailed results as Pandas DataFrame for compatibility
            pandas_df = df.toPandas()
            pandas_df.to_json("detailed_results.json", orient="records", indent=2)
            pandas_df.to_csv("results.csv", index=False)
            
            logger.info("✅ Results saved:")
            logger.info("   - output/spark_results/ (Spark DataFrame)")
            logger.info("   - pyspark_analytics.json (Analytics)")
            logger.info("   - detailed_results.json (Detailed data)")
            logger.info("   - results.csv (CSV format)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving results: {e}")
            return False
    
    def run_pipeline(self) -> bool:
        """Execute the complete integrated pipeline"""
        try:
            start_time = time.time()
            logger.info("🚀 Starting Integrated Reddit Sentiment Analysis Pipeline")
            logger.info("=" * 70)
            
            # Step 1: Initialize components
            if not self.initialize_spark():
                return False
            
            if not self.initialize_sentiment_model():
                return False
            
            # Step 2: Fetch Reddit data
            if not self.fetch_reddit_data():
                logger.error("❌ Failed to fetch Reddit data")
                return False
            
            # Step 3: Load data to Spark
            df = self.load_reddit_data_to_spark()
            if df is None:
                return False
            
            # Step 4: Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Step 5: Perform sentiment analysis
            df_with_sentiment = self.perform_sentiment_analysis(df_processed)
            
            # Step 6: Analyze results
            analytics = self.analyze_results(df_with_sentiment)
            
            # Step 7: Save results
            if not self.save_results(df_with_sentiment, analytics):
                return False
            
            # Step 8: Display summary
            execution_time = time.time() - start_time
            self._display_summary(analytics, execution_time)
            
            logger.info("🎉 Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return False
        finally:
            if self.spark:
                self.spark.stop()
    
    def _display_summary(self, analytics: Dict, execution_time: float):
        """Display pipeline summary"""
        print("\n" + "=" * 70)
        print("📊 INTEGRATED PYSPARK PIPELINE RESULTS")
        print("=" * 70)
        
        metadata = analytics.get("pipeline_metadata", {})
        print(f"Pipeline ID: {metadata.get('pipeline_id', 'N/A')}")
        print(f"Total Comments: {metadata.get('total_comments', 0)}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        sentiment_dist = analytics.get("sentiment_distribution", {})
        if sentiment_dist:
            print(f"\n🎭 SENTIMENT DISTRIBUTION:")
            for sentiment, count in sentiment_dist.items():
                percentage = (count / metadata.get('total_comments', 1)) * 100
                print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        confidence = analytics.get("average_confidence", {})
        if confidence:
            print(f"\n🎯 AVERAGE CONFIDENCE:")
            for sentiment, conf in confidence.items():
                print(f"  {sentiment}: {conf:.1f}%")
        
        text_stats = analytics.get("text_statistics", {})
        if text_stats:
            print(f"\n📝 TEXT STATISTICS:")
            print(f"  Average Length: {text_stats.get('avg_length', 0):.1f} characters")
            print(f"  Average Words: {text_stats.get('avg_words', 0):.1f}")
            print(f"  Average Score: {text_stats.get('avg_score', 0):.1f}")
        
        print("=" * 70)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Integrated PySpark Reddit Sentiment Analysis Pipeline")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--batch", action="store_true", help="Enable batch processing mode")
    
    args = parser.parse_args()
    
    # Load configuration
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override batch mode if specified
    if config and args.batch:
        config["reddit"]["batch_mode"] = True
    
    # Create and run pipeline
    pipeline = IntegratedRedditSentimentPipeline(config)
    success = pipeline.run_pipeline()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()