"""
PySpark Sentiment Processing Module for Reddit Comments
Performs distributed sentiment analysis using existing ML models with big data capabilities
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, TimestampType, ArrayType
)
from pyspark.sql.functions import (
    col, udf, broadcast, when, coalesce, lit, 
    current_timestamp, size, split, regexp_replace
)
import logging

# Add project root to path to import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing sentiment analysis components
try:
    from analyzer import Analyzer
    from arguments import args
except ImportError as e:
    print(f"Warning: Could not import existing modules: {e}")
    print("Make sure analyzer.py and arguments.py are in the same directory")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistributedSentimentProcessor:
    """
    PySpark-based distributed sentiment analysis processor.
    Integrates existing ML models with big data processing capabilities.
    """
    
    def __init__(self, app_name: str = "RedditSentimentProcessor", 
                 batch_size: int = 1000):
        """Initialize Spark session with ML-optimized configuration."""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "64MB") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", str(batch_size)) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.batch_size = batch_size
        
        # Initialize sentiment analyzer (will be broadcasted to workers)
        self.analyzer = None
        self._initialize_analyzer()
        
        logger.info(f"Initialized Distributed Sentiment Processor: {app_name}")
    
    def _initialize_analyzer(self):
        """Initialize the sentiment analyzer for distributed processing."""
        try:
            self.analyzer = Analyzer(will_train=False, args=args)
            logger.info("Sentiment analyzer initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize analyzer: {e}")
            logger.info("Will use fallback sentiment analysis method")
    
    def create_sentiment_schema(self) -> StructType:
        """Define schema for sentiment analysis results."""
        return StructType([
            StructField("text", StringType(), nullable=False),
            StructField("sentiment", StringType(), nullable=True),
            StructField("confidence", DoubleType(), nullable=True),
            StructField("positive_prob", DoubleType(), nullable=True),
            StructField("negative_prob", DoubleType(), nullable=True),
            StructField("neutral_prob", DoubleType(), nullable=True),
            StructField("processing_timestamp", TimestampType(), nullable=True),
            StructField("model_version", StringType(), nullable=True),
            StructField("text_length", IntegerType(), nullable=True),
            StructField("word_count", IntegerType(), nullable=True)
        ])
    
    def preprocess_text_distributed(self, df: DataFrame) -> DataFrame:
        """
        Apply distributed text preprocessing for sentiment analysis.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            DataFrame with preprocessed text
        """
        logger.info("Starting distributed text preprocessing")
        
        # Text cleaning and preprocessing
        df = df.withColumn("text_cleaned", 
                          regexp_replace(col("text"), r"http\S+|www\S+", "")) \
               .withColumn("text_cleaned", 
                          regexp_replace(col("text_cleaned"), r"@\w+|#\w+", "")) \
               .withColumn("text_cleaned", 
                          regexp_replace(col("text_cleaned"), r"[^a-zA-Z0-9\s]", "")) \
               .withColumn("text_cleaned", 
                          regexp_replace(col("text_cleaned"), r"\s+", " "))
        
        # Add text statistics
        df = df.withColumn("word_count", size(split(col("text_cleaned"), " "))) \
               .withColumn("text_length", col("comment_length"))
        
        # Filter out very short or empty texts
        df = df.filter((col("word_count") >= 2) & (col("text_length") >= 10))
        
        logger.info(f"Text preprocessing completed. Records: {df.count()}")
        return df
    
    def analyze_sentiment_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch sentiment analysis function for distributed processing.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for text in texts:
            try:
                if self.analyzer:
                    # Use existing ML model
                    sentiment, confidence = self.analyzer.classify_sentiment(text)
                    
                    # Extract probabilities (assuming your analyzer returns this)
                    result = {
                        "sentiment": sentiment.lower(),
                        "confidence": float(confidence) / 100.0,
                        "positive_prob": float(confidence) / 100.0 if sentiment.lower() == "positive" else (100.0 - float(confidence)) / 100.0,
                        "negative_prob": float(confidence) / 100.0 if sentiment.lower() == "negative" else (100.0 - float(confidence)) / 100.0,
                        "neutral_prob": 0.1,  # Placeholder - adjust based on your model
                        "model_version": "bert-sentiment-v1"
                    }
                else:
                    # Fallback simple sentiment analysis
                    result = self._simple_sentiment_analysis(text)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error analyzing text: {str(e)}")
                results.append({
                    "sentiment": "unknown",
                    "confidence": 0.0,
                    "positive_prob": 0.33,
                    "negative_prob": 0.33,
                    "neutral_prob": 0.34,
                    "model_version": "fallback"
                })
        
        return results
    
    def _simple_sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Fallback simple sentiment analysis using keyword matching."""
        positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'like', 'amazing', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst', 'sucks']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
            confidence = min(0.6 + (pos_count - neg_count) * 0.1, 0.9)
        elif neg_count > pos_count:
            sentiment = "negative"
            confidence = min(0.6 + (neg_count - pos_count) * 0.1, 0.9)
        else:
            sentiment = "neutral"
            confidence = 0.5
        
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_prob": confidence if sentiment == "positive" else (1 - confidence) / 2,
            "negative_prob": confidence if sentiment == "negative" else (1 - confidence) / 2,
            "neutral_prob": confidence if sentiment == "neutral" else (1 - confidence),
            "model_version": "simple-keyword"
        }
    
    def process_sentiment_distributed(self, df: DataFrame) -> DataFrame:
        """
        Perform distributed sentiment analysis on the input DataFrame.
        
        Args:
            df: Input DataFrame with preprocessed text
            
        Returns:
            DataFrame with sentiment analysis results
        """
        logger.info("Starting distributed sentiment analysis")
        
        # Create UDF for sentiment analysis
        def sentiment_udf(text_col):
            def analyze_text(text):
                if not text or text.strip() == "":
                    return {
                        "sentiment": "unknown",
                        "confidence": 0.0,
                        "positive_prob": 0.33,
                        "negative_prob": 0.33,
                        "neutral_prob": 0.34,
                        "model_version": "empty"
                    }
                
                try:
                    if self.analyzer:
                        sentiment, confidence = self.analyzer.classify_sentiment(text)
                        return {
                            "sentiment": sentiment.lower(),
                            "confidence": float(confidence) / 100.0,
                            "positive_prob": float(confidence) / 100.0 if sentiment.lower() == "positive" else (100.0 - float(confidence)) / 100.0,
                            "negative_prob": float(confidence) / 100.0 if sentiment.lower() == "negative" else (100.0 - float(confidence)) / 100.0,
                            "neutral_prob": 0.1,
                            "model_version": "bert-sentiment-v1"
                        }
                    else:
                        return self._simple_sentiment_analysis(text)
                except:
                    return self._simple_sentiment_analysis(text)
            
            return udf(analyze_text, 
                      StructType([
                          StructField("sentiment", StringType(), True),
                          StructField("confidence", DoubleType(), True),
                          StructField("positive_prob", DoubleType(), True),
                          StructField("negative_prob", DoubleType(), True),
                          StructField("neutral_prob", DoubleType(), True),
                          StructField("model_version", StringType(), True)
                      ]))
        
        # Apply sentiment analysis
        sentiment_analyzer = sentiment_udf(col("text_cleaned"))
        
        df_with_sentiment = df.withColumn("sentiment_result", sentiment_analyzer(col("text_cleaned"))) \
                             .withColumn("sentiment", col("sentiment_result.sentiment")) \
                             .withColumn("confidence", col("sentiment_result.confidence")) \
                             .withColumn("positive_prob", col("sentiment_result.positive_prob")) \
                             .withColumn("negative_prob", col("sentiment_result.negative_prob")) \
                             .withColumn("neutral_prob", col("sentiment_result.neutral_prob")) \
                             .withColumn("model_version", col("sentiment_result.model_version")) \
                             .withColumn("processing_timestamp", current_timestamp()) \
                             .drop("sentiment_result")
        
        logger.info(f"Sentiment analysis completed for {df_with_sentiment.count()} records")
        return df_with_sentiment
    
    def create_sentiment_summary(self, df: DataFrame) -> Dict[str, Any]:
        """Generate comprehensive sentiment analysis summary."""
        logger.info("Generating sentiment analysis summary")
        
        # Basic sentiment counts
        sentiment_counts = df.groupBy("sentiment").count().collect()
        sentiment_stats = {row["sentiment"]: row["count"] for row in sentiment_counts}
        
        # Confidence statistics
        confidence_stats = df.select("confidence").describe().collect()
        confidence_summary = {row["summary"]: float(row["confidence"]) for row in confidence_stats}
        
        # Model version statistics
        model_stats = df.groupBy("model_version").count().collect()
        model_summary = {row["model_version"]: row["count"] for row in model_stats}
        
        # Text length vs sentiment correlation
        avg_sentiment_by_length = df.groupBy("sentiment") \
                                   .agg({"text_length": "avg", "word_count": "avg"}) \
                                   .collect()
        
        length_correlation = {}
        for row in avg_sentiment_by_length:
            length_correlation[row["sentiment"]] = {
                "avg_text_length": float(row["avg(text_length)"]),
                "avg_word_count": float(row["avg(word_count)"])
            }
        
        summary = {
            "total_processed": df.count(),
            "sentiment_distribution": sentiment_stats,
            "confidence_statistics": confidence_summary,
            "model_usage": model_summary,
            "length_sentiment_correlation": length_correlation,
            "processing_timestamp": str(current_timestamp())
        }
        
        logger.info(f"Summary generated: {summary}")
        return summary
    
    def save_sentiment_results(self, df: DataFrame, output_path: str, 
                              include_original: bool = True) -> None:
        """
        Save sentiment analysis results in optimized format.
        
        Args:
            df: DataFrame with sentiment results
            output_path: Output path for results
            include_original: Whether to include original text in output
        """
        logger.info(f"Saving sentiment results to: {output_path}")
        
        # Select relevant columns for output
        if include_original:
            output_cols = [
                "text", "text_cleaned", "sentiment", "confidence",
                "positive_prob", "negative_prob", "neutral_prob",
                "text_length", "word_count", "model_version", "processing_timestamp"
            ]
        else:
            output_cols = [
                "sentiment", "confidence", "positive_prob", "negative_prob", 
                "neutral_prob", "text_length", "word_count", "model_version", 
                "processing_timestamp"
            ]
        
        # Select available columns
        available_cols = [col for col in output_cols if col in df.columns]
        result_df = df.select(*available_cols)
        
        # Save as partitioned parquet for optimal performance
        result_df.write \
                 .mode("overwrite") \
                 .partitionBy("sentiment") \
                 .parquet(output_path)
        
        logger.info("Sentiment results saved successfully")
    
    def process_streaming_sentiment(self, input_path: str, output_path: str, 
                                  checkpoint_path: str) -> None:
        """
        Process streaming sentiment analysis for real-time Reddit comments.
        
        Args:
            input_path: Path to monitor for new comment files
            output_path: Path to write streaming results
            checkpoint_path: Checkpoint location for streaming
        """
        logger.info("Starting streaming sentiment analysis")
        
        # Read streaming data
        streaming_df = self.spark.readStream \
                                .schema(self.create_sentiment_schema()) \
                                .json(input_path)
        
        # Apply sentiment processing
        processed_df = self.preprocess_text_distributed(streaming_df)
        sentiment_df = self.process_sentiment_distributed(processed_df)
        
        # Write streaming output
        query = sentiment_df.writeStream \
                           .outputMode("append") \
                           .format("parquet") \
                           .option("path", output_path) \
                           .option("checkpointLocation", checkpoint_path) \
                           .partitionBy("sentiment") \
                           .start()
        
        logger.info("Streaming sentiment analysis started")
        return query
    
    def stop_session(self):
        """Stop Spark session and cleanup resources."""
        self.spark.stop()
        logger.info("Sentiment processor session stopped")


def main():
    """Example usage of the Distributed Sentiment Processor."""
    processor = DistributedSentimentProcessor("RedditSentimentProcessor")
    
    try:
        # Read processed comments (assuming from ingestion pipeline)
        input_path = "data/processed_comments"
        if os.path.exists(input_path):
            df = processor.spark.read.parquet(input_path)
        else:
            # Fallback to JSON file
            df = processor.spark.read.json("comments.json")
            df = df.select("text").withColumn("comment_length", col("text").cast("int"))
        
        # Preprocess text
        preprocessed_df = processor.preprocess_text_distributed(df)
        
        # Perform sentiment analysis
        sentiment_df = processor.process_sentiment_distributed(preprocessed_df)
        
        # Generate summary
        summary = processor.create_sentiment_summary(sentiment_df)
        print(f"\nSentiment Analysis Summary:\n{json.dumps(summary, indent=2)}")
        
        # Save results
        processor.save_sentiment_results(sentiment_df, "data/sentiment_results")
        
        # Show sample results
        print("\nSample Sentiment Results:")
        sentiment_df.select("text", "sentiment", "confidence", "positive_prob") \
                   .show(10, truncate=False)
        
    except Exception as e:
        logger.error(f"Sentiment processing failed: {str(e)}")
    finally:
        processor.stop_session()


if __name__ == "__main__":
    main()