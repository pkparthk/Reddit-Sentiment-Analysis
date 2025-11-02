"""
PySpark Data Ingestion Module for Reddit Sentiment Analysis
Handles batch processing and ingestion of Reddit comments for big data analytics
"""

import os
import json
from typing import List, Dict, Any
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    DoubleType, TimestampType, ArrayType
)
from pyspark.sql.functions import (
    col, when, isnan, isnull, length, trim, regexp_replace,
    current_timestamp, monotonically_increasing_id, lit
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditDataIngestion:
    """
    PySpark-based data ingestion pipeline for Reddit comments.
    Provides distributed processing capabilities for large-scale comment analysis.
    """
    
    def __init__(self, app_name: str = "RedditSentimentIngestion"):
        """Initialize Spark session with optimized configuration for big data processing."""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .getOrCreate()
        
        # Set log level to reduce verbosity
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Initialized Spark session: {app_name}")
        
    def define_comment_schema(self) -> StructType:
        """Define schema for Reddit comments to ensure data quality and performance."""
        return StructType([
            StructField("id", StringType(), nullable=True),
            StructField("author", StringType(), nullable=True),
            StructField("text", StringType(), nullable=False),
            StructField("score", IntegerType(), nullable=True),
            StructField("depth", IntegerType(), nullable=True),
            StructField("created_utc", TimestampType(), nullable=True),
            StructField("subreddit", StringType(), nullable=True),
            StructField("post_id", StringType(), nullable=True),
            StructField("processing_timestamp", TimestampType(), nullable=True),
            StructField("comment_length", IntegerType(), nullable=True),
            StructField("is_valid", StringType(), nullable=True)
        ])
    
    def ingest_jsonl_comments(self, file_path: str) -> DataFrame:
        """
        Ingest Reddit comments from JSONL file using PySpark DataFrame API.
        Supports distributed processing of large comment datasets.
        
        Args:
            file_path: Path to the JSONL file containing Reddit comments
            
        Returns:
            PySpark DataFrame with processed comments
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Comments file not found: {file_path}")
            
        logger.info(f"Starting ingestion from: {file_path}")
        
        try:
            # Read JSONL file with schema inference
            df = self.spark.read.json(file_path, schema=self.define_comment_schema())
            
            # If schema doesn't match, read without schema and process
            if df.count() == 0:
                df = self.spark.read.text(file_path)
                df = self._process_text_to_json(df)
            
            # Add processing metadata and data quality checks
            df = self._add_processing_metadata(df)
            df = self._apply_data_quality_checks(df)
            
            logger.info(f"Successfully ingested {df.count()} comments")
            return df
            
        except Exception as e:
            logger.error(f"Error during ingestion: {str(e)}")
            raise
    
    def ingest_batch_files(self, directory_path: str, file_pattern: str = "*.json") -> DataFrame:
        """
        Ingest multiple comment files for batch processing.
        Ideal for processing large volumes of Reddit comments.
        
        Args:
            directory_path: Directory containing comment files
            file_pattern: File pattern to match (default: *.json)
            
        Returns:
            Unified PySpark DataFrame with all comments
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        logger.info(f"Starting batch ingestion from: {directory_path}")
        
        try:
            # Read all matching files in the directory
            file_path = os.path.join(directory_path, file_pattern)
            df = self.spark.read.json(file_path, schema=self.define_comment_schema())
            
            # Add batch processing metadata
            df = df.withColumn("batch_id", lit(f"batch_{current_timestamp()}"))
            df = self._add_processing_metadata(df)
            df = self._apply_data_quality_checks(df)
            
            logger.info(f"Successfully processed batch with {df.count()} total comments")
            return df
            
        except Exception as e:
            logger.error(f"Error during batch ingestion: {str(e)}")
            raise
    
    def _process_text_to_json(self, text_df: DataFrame) -> DataFrame:
        """Process raw text lines to structured JSON format."""
        from pyspark.sql.functions import from_json
        
        # Define simple schema for text processing
        json_schema = StructType([
            StructField("text", StringType(), nullable=True)
        ])
        
        # Parse JSON from text lines
        df = text_df.select(
            from_json(col("value"), json_schema).alias("data")
        ).select("data.*")
        
        return df
    
    def _add_processing_metadata(self, df: DataFrame) -> DataFrame:
        """Add metadata for tracking and analytics."""
        df = df.withColumn("processing_timestamp", current_timestamp()) \
               .withColumn("record_id", monotonically_increasing_id()) \
               .withColumn("comment_length", length(col("text")))
        
        return df
    
    def _apply_data_quality_checks(self, df: DataFrame) -> DataFrame:
        """Apply data quality checks and validation rules."""
        # Clean and validate text data
        df = df.withColumn("text", 
                          when(col("text").isNull() | (col("text") == ""), None)
                          .otherwise(trim(regexp_replace(col("text"), r"[\r\n\t]+", " "))))
        
        # Mark invalid records
        df = df.withColumn("is_valid",
                          when(
                              (col("text").isNull()) |
                              (length(col("text")) < 3) |
                              (col("text").contains("[deleted]")) |
                              (col("text").contains("[removed]")),
                              "invalid"
                          ).otherwise("valid"))
        
        return df
    
    def filter_valid_comments(self, df: DataFrame) -> DataFrame:
        """Filter out invalid or low-quality comments."""
        valid_df = df.filter(col("is_valid") == "valid")
        
        total_count = df.count()
        valid_count = valid_df.count()
        
        logger.info(f"Filtered comments: {valid_count}/{total_count} valid comments retained")
        return valid_df
    
    def create_partitioned_dataset(self, df: DataFrame, output_path: str, 
                                 partition_cols: List[str] = None) -> None:
        """
        Create partitioned dataset for optimized big data processing.
        
        Args:
            df: Input DataFrame
            output_path: Path to save partitioned dataset
            partition_cols: Columns to partition by (default: processing date)
        """
        if partition_cols is None:
            # Add date partition column
            from pyspark.sql.functions import date_format
            df = df.withColumn("processing_date", 
                             date_format(col("processing_timestamp"), "yyyy-MM-dd"))
            partition_cols = ["processing_date"]
        
        logger.info(f"Creating partitioned dataset at: {output_path}")
        
        # Write partitioned parquet files for optimal performance
        df.write \
          .mode("overwrite") \
          .partitionBy(partition_cols) \
          .parquet(output_path)
        
        logger.info("Partitioned dataset created successfully")
    
    def get_ingestion_statistics(self, df: DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistics about ingested data."""
        stats = {}
        
        # Basic counts
        stats["total_records"] = df.count()
        stats["valid_records"] = df.filter(col("is_valid") == "valid").count()
        
        # Text length statistics
        text_stats = df.select("comment_length").describe()
        stats["text_length_stats"] = {row["summary"]: row["comment_length"] 
                                    for row in text_stats.collect()}
        
        # Author statistics
        stats["unique_authors"] = df.select("author").distinct().count()
        
        # Score statistics if available
        if "score" in df.columns:
            score_stats = df.select("score").describe()
            stats["score_stats"] = {row["summary"]: row["score"] 
                                  for row in score_stats.collect()}
        
        logger.info(f"Ingestion statistics: {stats}")
        return stats
    
    def save_processed_data(self, df: DataFrame, output_path: str, 
                          format_type: str = "parquet") -> None:
        """
        Save processed data in optimized format for downstream processing.
        
        Args:
            df: Processed DataFrame
            output_path: Output file path
            format_type: Output format (parquet, delta, json)
        """
        logger.info(f"Saving processed data to: {output_path} (format: {format_type})")
        
        writer = df.write.mode("overwrite")
        
        if format_type.lower() == "parquet":
            writer.parquet(output_path)
        elif format_type.lower() == "delta":
            writer.format("delta").save(output_path)
        elif format_type.lower() == "json":
            writer.json(output_path)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        logger.info("Data saved successfully")
    
    def stop_session(self):
        """Stop the Spark session and cleanup resources."""
        self.spark.stop()
        logger.info("Spark session stopped")


def main():
    """Example usage of the Reddit Data Ingestion pipeline."""
    ingestion = RedditDataIngestion("RedditSentimentIngestion")
    
    try:
        # Ingest comments from JSONL file
        df = ingestion.ingest_jsonl_comments("comments.json")
        
        # Apply data quality filters
        clean_df = ingestion.filter_valid_comments(df)
        
        # Generate statistics
        stats = ingestion.get_ingestion_statistics(clean_df)
        print(f"Processing Statistics: {stats}")
        
        # Save processed data for sentiment analysis
        ingestion.save_processed_data(clean_df, "data/processed_comments", "parquet")
        
        # Show sample data
        print("\nSample processed comments:")
        clean_df.select("text", "comment_length", "is_valid").show(5, truncate=False)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
    finally:
        ingestion.stop_session()


if __name__ == "__main__":
    main()