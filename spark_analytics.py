"""
PySpark Analytics Pipeline for Reddit Sentiment Analysis
Provides comprehensive big data analytics, insights, and reporting capabilities
"""

import os
import json
import math
from typing import List, Dict, Any, Tuple
from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    IntegerType, TimestampType, ArrayType
)
from pyspark.sql.functions import (
    col, count, avg, sum as spark_sum, max as spark_max, min as spark_min,
    stddev, var_pop, percentile_approx, collect_list, collect_set,
    desc, asc, when, coalesce, lit, current_timestamp,
    date_format, hour, dayofweek, month, year,
    regexp_extract, split, size, length,
    lag, lead, row_number, rank, dense_rank,
    ntile, percent_rank, cume_dist,
    corr, covar_pop, approx_count_distinct
)
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available, some visualizations disabled")

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml.clustering import KMeans
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RedditSentimentAnalytics:
    """
    Advanced analytics pipeline for Reddit sentiment analysis using PySpark.
    Provides comprehensive insights, trends, and statistical analysis.
    """
    
    def __init__(self, app_name: str = "RedditSentimentAnalytics"):
        """Initialize Spark session with analytics-optimized configuration."""
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128MB") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
            .config("spark.sql.adaptive.skewJoin.enabled", "true") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        logger.info(f"Initialized Reddit Sentiment Analytics: {app_name}")
    
    def load_sentiment_data(self, input_path: str) -> DataFrame:
        """Load processed sentiment data for analytics."""
        logger.info(f"Loading sentiment data from: {input_path}")
        
        try:
            if os.path.exists(input_path) and os.path.isdir(input_path):
                # Load partitioned parquet data
                df = self.spark.read.parquet(input_path)
            else:
                # Fallback to JSON
                df = self.spark.read.json(input_path)
            
            logger.info(f"Loaded {df.count()} sentiment records")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def basic_sentiment_statistics(self, df: DataFrame) -> Dict[str, Any]:
        """Generate comprehensive basic statistics for sentiment analysis."""
        logger.info("Generating basic sentiment statistics")
        
        stats = {}
        
        # Total records
        total_records = df.count()
        stats["total_records"] = total_records
        
        # Sentiment distribution
        sentiment_dist = df.groupBy("sentiment") \
                          .agg(count("*").alias("count"),
                               (count("*") * 100.0 / total_records).alias("percentage")) \
                          .orderBy(desc("count")) \
                          .collect()
        
        stats["sentiment_distribution"] = {
            row["sentiment"]: {
                "count": row["count"],
                "percentage": round(row["percentage"], 2)
            } for row in sentiment_dist
        }
        
        # Confidence statistics
        confidence_stats = df.select(
            avg("confidence").alias("avg_confidence"),
            spark_min("confidence").alias("min_confidence"),
            spark_max("confidence").alias("max_confidence"),
            stddev("confidence").alias("stddev_confidence"),
            percentile_approx("confidence", 0.5).alias("median_confidence"),
            percentile_approx("confidence", 0.25).alias("q1_confidence"),
            percentile_approx("confidence", 0.75).alias("q3_confidence")
        ).collect()[0]
        
        stats["confidence_statistics"] = {
            "average": round(confidence_stats["avg_confidence"], 4),
            "minimum": round(confidence_stats["min_confidence"], 4),
            "maximum": round(confidence_stats["max_confidence"], 4),
            "std_deviation": round(confidence_stats["stddev_confidence"], 4),
            "median": round(confidence_stats["median_confidence"], 4),
            "q1": round(confidence_stats["q1_confidence"], 4),
            "q3": round(confidence_stats["q3_confidence"], 4)
        }
        
        # Text length analysis
        if "text_length" in df.columns:
            length_stats = df.select(
                avg("text_length").alias("avg_length"),
                spark_min("text_length").alias("min_length"),
                spark_max("text_length").alias("max_length"),
                stddev("text_length").alias("stddev_length")
            ).collect()[0]
            
            stats["text_length_statistics"] = {
                "average": round(length_stats["avg_length"], 2),
                "minimum": length_stats["min_length"],
                "maximum": length_stats["max_length"],
                "std_deviation": round(length_stats["stddev_length"], 2)
            }
        
        logger.info(f"Basic statistics completed: {stats}")
        return stats
    
    def sentiment_confidence_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze relationship between sentiment and confidence levels."""
        logger.info("Analyzing sentiment vs confidence relationships")
        
        # Confidence ranges analysis
        confidence_ranges = df.withColumn("confidence_range",
                                        when(col("confidence") >= 0.8, "High (0.8+)")
                                        .when(col("confidence") >= 0.6, "Medium (0.6-0.8)")
                                        .when(col("confidence") >= 0.4, "Low (0.4-0.6)")
                                        .otherwise("Very Low (<0.4)"))
        
        range_dist = confidence_ranges.groupBy("sentiment", "confidence_range") \
                                    .count() \
                                    .orderBy("sentiment", desc("count")) \
                                    .collect()
        
        confidence_analysis = {}
        for row in range_dist:
            sentiment = row["sentiment"]
            if sentiment not in confidence_analysis:
                confidence_analysis[sentiment] = {}
            confidence_analysis[sentiment][row["confidence_range"]] = row["count"]
        
        # Average confidence by sentiment
        avg_confidence_by_sentiment = df.groupBy("sentiment") \
                                       .agg(avg("confidence").alias("avg_confidence"),
                                           count("*").alias("count")) \
                                       .collect()
        
        sentiment_confidence = {
            row["sentiment"]: {
                "average_confidence": round(row["avg_confidence"], 4),
                "sample_size": row["count"]
            } for row in avg_confidence_by_sentiment
        }
        
        return {
            "confidence_range_distribution": confidence_analysis,
            "average_confidence_by_sentiment": sentiment_confidence
        }
    
    def text_length_sentiment_correlation(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze correlation between text characteristics and sentiment."""
        logger.info("Analyzing text characteristics vs sentiment")
        
        if "text_length" not in df.columns or "word_count" not in df.columns:
            logger.warning("Text length or word count columns not found")
            return {"error": "Required columns not found"}
        
        # Length distribution by sentiment
        length_by_sentiment = df.groupBy("sentiment") \
                               .agg(avg("text_length").alias("avg_text_length"),
                                   avg("word_count").alias("avg_word_count"),
                                   spark_min("text_length").alias("min_text_length"),
                                   spark_max("text_length").alias("max_text_length"),
                                   stddev("text_length").alias("stddev_text_length")) \
                               .collect()
        
        length_analysis = {}
        for row in length_by_sentiment:
            sentiment = row["sentiment"]
            length_analysis[sentiment] = {
                "avg_text_length": round(row["avg_text_length"], 2),
                "avg_word_count": round(row["avg_word_count"], 2),
                "min_text_length": row["min_text_length"],
                "max_text_length": row["max_text_length"],
                "stddev_text_length": round(row["stddev_text_length"], 2)
            }
        
        # Correlation analysis
        correlation_stats = df.select(
            corr("text_length", "confidence").alias("length_confidence_corr"),
            corr("word_count", "confidence").alias("words_confidence_corr")
        ).collect()[0]
        
        return {
            "length_by_sentiment": length_analysis,
            "correlations": {
                "text_length_confidence": round(correlation_stats["length_confidence_corr"], 4),
                "word_count_confidence": round(correlation_stats["words_confidence_corr"], 4)
            }
        }
    
    def temporal_sentiment_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Perform temporal analysis of sentiment patterns."""
        logger.info("Performing temporal sentiment analysis")
        
        if "processing_timestamp" not in df.columns:
            logger.warning("Timestamp column not found, skipping temporal analysis")
            return {"error": "Timestamp column not found"}
        
        # Add time-based columns
        temporal_df = df.withColumn("hour", hour("processing_timestamp")) \
                       .withColumn("day_of_week", dayofweek("processing_timestamp")) \
                       .withColumn("month", month("processing_timestamp")) \
                       .withColumn("year", year("processing_timestamp"))
        
        # Hourly sentiment distribution
        hourly_sentiment = temporal_df.groupBy("hour", "sentiment") \
                                    .count() \
                                    .orderBy("hour", "sentiment") \
                                    .collect()
        
        hourly_analysis = {}
        for row in hourly_sentiment:
            hour = row["hour"]
            if hour not in hourly_analysis:
                hourly_analysis[hour] = {}
            hourly_analysis[hour][row["sentiment"]] = row["count"]
        
        # Daily sentiment trends
        daily_sentiment = temporal_df.groupBy("day_of_week", "sentiment") \
                                   .count() \
                                   .orderBy("day_of_week", "sentiment") \
                                   .collect()
        
        daily_analysis = {}
        for row in daily_sentiment:
            day = row["day_of_week"]
            if day not in daily_analysis:
                daily_analysis[day] = {}
            daily_analysis[day][row["sentiment"]] = row["count"]
        
        return {
            "hourly_sentiment_distribution": hourly_analysis,
            "daily_sentiment_distribution": daily_analysis
        }
    
    def advanced_sentiment_clustering(self, df: DataFrame) -> Dict[str, Any]:
        """Perform clustering analysis on sentiment data."""
        logger.info("Performing sentiment clustering analysis")
        
        # Prepare features for clustering
        feature_cols = ["confidence", "positive_prob", "negative_prob", "neutral_prob"]
        available_features = [col for col in feature_cols if col in df.columns]
        
        if len(available_features) < 2:
            logger.warning("Insufficient features for clustering analysis")
            return {"error": "Insufficient features"}
        
        # Create feature vector
        assembler = VectorAssembler(inputCols=available_features, outputCol="features")
        feature_df = assembler.transform(df)
        
        # Scale features
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
        scaler_model = scaler.fit(feature_df)
        scaled_df = scaler_model.transform(feature_df)
        
        # Perform K-means clustering
        kmeans = KMeans(k=3, featuresCol="scaled_features", predictionCol="cluster")
        kmeans_model = kmeans.fit(scaled_df)
        clustered_df = kmeans_model.transform(scaled_df)
        
        # Analyze clusters
        cluster_analysis = clustered_df.groupBy("cluster", "sentiment") \
                                     .count() \
                                     .orderBy("cluster", "sentiment") \
                                     .collect()
        
        cluster_stats = {}
        for row in cluster_analysis:
            cluster_id = row["cluster"]
            if cluster_id not in cluster_stats:
                cluster_stats[cluster_id] = {}
            cluster_stats[cluster_id][row["sentiment"]] = row["count"]
        
        # Cluster centers
        centers = kmeans_model.clusterCenters()
        cluster_centers = {
            i: {feature: float(center[j]) for j, feature in enumerate(available_features)}
            for i, center in enumerate(centers)
        }
        
        return {
            "cluster_sentiment_distribution": cluster_stats,
            "cluster_centers": cluster_centers,
            "features_used": available_features
        }
    
    def model_performance_analysis(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze performance characteristics of sentiment models."""
        logger.info("Analyzing model performance")
        
        if "model_version" not in df.columns:
            logger.warning("Model version column not found")
            return {"error": "Model version not found"}
        
        # Model usage statistics
        model_stats = df.groupBy("model_version") \
                       .agg(count("*").alias("usage_count"),
                           avg("confidence").alias("avg_confidence"),
                           stddev("confidence").alias("confidence_stddev")) \
                       .collect()
        
        model_analysis = {}
        for row in model_stats:
            model_version = row["model_version"]
            model_analysis[model_version] = {
                "usage_count": row["usage_count"],
                "average_confidence": round(row["avg_confidence"], 4),
                "confidence_stddev": round(row["confidence_stddev"], 4) if row["confidence_stddev"] else 0
            }
        
        # Model performance by sentiment
        model_sentiment_stats = df.groupBy("model_version", "sentiment") \
                                 .agg(count("*").alias("count"),
                                     avg("confidence").alias("avg_confidence")) \
                                 .collect()
        
        model_sentiment_analysis = {}
        for row in model_sentiment_stats:
            model = row["model_version"]
            sentiment = row["sentiment"]
            if model not in model_sentiment_analysis:
                model_sentiment_analysis[model] = {}
            model_sentiment_analysis[model][sentiment] = {
                "count": row["count"],
                "avg_confidence": round(row["avg_confidence"], 4)
            }
        
        return {
            "model_usage_statistics": model_analysis,
            "model_sentiment_performance": model_sentiment_analysis
        }
    
    def generate_comprehensive_report(self, df: DataFrame) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        logger.info("Generating comprehensive analytics report")
        
        report = {
            "metadata": {
                "report_timestamp": str(current_timestamp()),
                "total_records": df.count(),
                "data_columns": df.columns
            }
        }
        
        try:
            # Basic statistics
            report["basic_statistics"] = self.basic_sentiment_statistics(df)
            
            # Confidence analysis
            report["confidence_analysis"] = self.sentiment_confidence_analysis(df)
            
            # Text characteristics
            report["text_analysis"] = self.text_length_sentiment_correlation(df)
            
            # Temporal analysis
            report["temporal_analysis"] = self.temporal_sentiment_analysis(df)
            
            # Clustering analysis
            report["clustering_analysis"] = self.advanced_sentiment_clustering(df)
            
            # Model performance
            report["model_performance"] = self.model_performance_analysis(df)
            
        except Exception as e:
            logger.error(f"Error generating report section: {str(e)}")
            report["errors"] = str(e)
        
        logger.info("Comprehensive report generated successfully")
        return report
    
    def save_analytics_results(self, results: Dict[str, Any], output_path: str):
        """Save analytics results to JSON file."""
        logger.info(f"Saving analytics results to: {output_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Analytics results saved successfully")
    
    def create_sentiment_dashboard_data(self, df: DataFrame) -> Dict[str, Any]:
        """Prepare data for dashboard visualization."""
        logger.info("Preparing dashboard data")
        
        # Top-level metrics
        total_comments = df.count()
        avg_confidence = df.select(avg("confidence")).collect()[0][0]
        
        # Sentiment pie chart data
        sentiment_counts = df.groupBy("sentiment").count().collect()
        pie_data = [{"sentiment": row["sentiment"], "count": row["count"]} 
                   for row in sentiment_counts]
        
        # Confidence histogram data
        confidence_ranges = df.withColumn("confidence_bucket",
                                        (col("confidence") * 10).cast("int") / 10.0) \
                             .groupBy("confidence_bucket") \
                             .count() \
                             .orderBy("confidence_bucket") \
                             .collect()
        
        histogram_data = [{"bucket": row["confidence_bucket"], "count": row["count"]} 
                         for row in confidence_ranges]
        
        # Recent trends (if timestamp available)
        trends_data = []
        if "processing_timestamp" in df.columns:
            recent_trends = df.withColumn("hour", hour("processing_timestamp")) \
                             .groupBy("hour", "sentiment") \
                             .count() \
                             .orderBy("hour") \
                             .collect()
            
            trends_data = [{"hour": row["hour"], "sentiment": row["sentiment"], "count": row["count"]} 
                          for row in recent_trends]
        
        dashboard_data = {
            "summary_metrics": {
                "total_comments": total_comments,
                "average_confidence": round(avg_confidence, 4) if avg_confidence else 0
            },
            "sentiment_distribution": pie_data,
            "confidence_histogram": histogram_data,
            "hourly_trends": trends_data
        }
        
        logger.info("Dashboard data prepared successfully")
        return dashboard_data
    
    def stop_session(self):
        """Stop Spark session and cleanup resources."""
        self.spark.stop()
        logger.info("Analytics session stopped")


def main():
    """Example usage of the Reddit Sentiment Analytics pipeline."""
    analytics = RedditSentimentAnalytics("RedditSentimentAnalytics")
    
    try:
        # Load sentiment data
        input_path = "data/sentiment_results"
        if not os.path.exists(input_path):
            logger.warning(f"Input path not found: {input_path}")
            return
        
        df = analytics.load_sentiment_data(input_path)
        
        # Generate comprehensive report
        report = analytics.generate_comprehensive_report(df)
        
        # Save analytics results
        analytics.save_analytics_results(report, "data/analytics_report.json")
        
        # Create dashboard data
        dashboard_data = analytics.create_sentiment_dashboard_data(df)
        analytics.save_analytics_results(dashboard_data, "data/dashboard_data.json")
        
        # Print key insights
        print("\n=== SENTIMENT ANALYSIS INSIGHTS ===")
        if "basic_statistics" in report:
            basic_stats = report["basic_statistics"]
            print(f"Total Comments Analyzed: {basic_stats.get('total_records', 0)}")
            print(f"Sentiment Distribution: {basic_stats.get('sentiment_distribution', {})}")
            
        if "confidence_analysis" in report:
            conf_stats = report["confidence_analysis"]["average_confidence_by_sentiment"]
            print(f"Average Confidence by Sentiment: {conf_stats}")
        
        print("\nFull analytics report saved to: data/analytics_report.json")
        print("Dashboard data saved to: data/dashboard_data.json")
        
    except Exception as e:
        logger.error(f"Analytics pipeline failed: {str(e)}")
    finally:
        analytics.stop_session()


if __name__ == "__main__":
    main()