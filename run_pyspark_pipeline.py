#!/usr/bin/env python3
"""
PySpark Reddit Sentiment Analysis Pipeline Runner
Complete big data processing pipeline with distributed computing capabilities
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime


def setup_spark_environment():
    """Setup Spark environment variables and configuration"""
    print("[SPARK] Setting up Spark environment...")
    
    # Try to find Java installation
    java_paths = [
        r"C:\Program Files\Java\jdk-11.0.16\bin\java.exe",
        r"C:\Program Files\Java\jdk-8u351-b10\bin\java.exe",
        r"C:\Program Files\OpenJDK\jdk-11.0.16\bin\java.exe",
        r"C:\Program Files (x86)\Java\jre1.8.0_351\bin\java.exe"
    ]
    
    java_home = None
    for java_path in java_paths:
        if os.path.exists(java_path):
            java_home = os.path.dirname(os.path.dirname(java_path))
            break
    
    if not java_home:
        print("[WARN] Java not found in common locations. Trying system PATH...")
        try:
            result = subprocess.run("java -version", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                print("[OK] Java found in system PATH")
            else:
                print("[ERROR] Java not found. Please install Java 8 or 11")
                return False
        except:
            print("[ERROR] Java not accessible")
            return False
    else:
        os.environ["JAVA_HOME"] = java_home
        print(f"[OK] JAVA_HOME set to: {java_home}")
    
    # Set other Spark environment variables
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    
    return True


def test_pyspark_installation():
    """Test if PySpark is properly installed and configured"""
    print("[SPARK] Testing PySpark installation...")
    
    try:
        from pyspark.sql import SparkSession
        
        # Try to create a minimal Spark session
        spark = SparkSession.builder \
            .appName("SparkTest") \
            .master("local[1]") \
            .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
            .getOrCreate()
        
        # Test basic functionality
        test_data = [("test", 1), ("spark", 2)]
        df = spark.createDataFrame(test_data, ["text", "value"])
        count = df.count()
        
        spark.stop()
        
        print(f"[OK] PySpark working correctly. Test dataframe has {count} rows")
        return True
        
    except Exception as e:
        print(f"[ERROR] PySpark test failed: {str(e)}")
        return False


def run_reddit_fetch():
    """Fetch Reddit comments"""
    print("[PIPELINE] Fetching fresh Reddit comments...")
    
    try:
        result = subprocess.run(
            "node fetch.js --mode=single",
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists("comments.json"):
            # Count comments
            comment_count = 0
            with open("comments.json", 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        comment_count += 1
            
            print(f"[OK] Fetched {comment_count} Reddit comments")
            return True
        else:
            print(f"[ERROR] Reddit fetch failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Reddit fetch error: {str(e)}")
        return False


def load_pipeline_config():
    """Load pipeline configuration"""
    try:
        with open("pipeline_config.json", 'r') as f:
            config = json.load(f)
        print("[OK] Pipeline configuration loaded")
        return config
    except Exception as e:
        print(f"[ERROR] Failed to load config: {str(e)}")
        # Return default config
        return {
            "spark": {
                "app_name": "RedditSentimentPipeline",
                "master": "local[*]",
                "executor_memory": "2g"
            }
        }


def create_data_directories():
    """Create necessary data directories"""
    directories = [
        "data",
        "data/processed_comments", 
        "data/sentiment_results",
        "data/analytics"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"[CREATED] Directory: {directory}")


def run_pyspark_pipeline():
    """Run the complete PySpark sentiment analysis pipeline"""
    print("[PIPELINE] Starting PySpark sentiment analysis pipeline...")
    
    try:
        # Import PySpark pipeline
        from main_pyspark import RedditSentimentPipeline
        
        # Load configuration
        config = load_pipeline_config()
        
        # Create pipeline instance
        pipeline = RedditSentimentPipeline(config)
        
        # Run complete pipeline
        print("[PIPELINE] Executing distributed processing...")
        results = pipeline.run_complete_pipeline()
        
        if results:
            print("[OK] PySpark pipeline completed successfully")
            return results
        else:
            print("[ERROR] PySpark pipeline returned no results")
            return None
            
    except Exception as e:
        print(f"[ERROR] PySpark pipeline failed: {str(e)}")
        print("[FALLBACK] Falling back to standard processing...")
        return run_standard_fallback()


def run_standard_fallback():
    """Fallback to standard processing if PySpark fails"""
    print("[FALLBACK] Running standard sentiment analysis...")
    
    try:
        # Use the working standard pipeline
        result = subprocess.run(
            "python run_pipeline.py",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("[OK] Standard pipeline completed successfully")
            
            # Load results if available
            if os.path.exists("reddit_sentiment_analysis.json"):
                with open("reddit_sentiment_analysis.json", 'r') as f:
                    return json.load(f)
            else:
                return {"status": "completed", "method": "standard"}
        else:
            print(f"[ERROR] Standard pipeline failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Standard fallback failed: {str(e)}")
        return None


def display_comprehensive_results(results):
    """Display comprehensive pipeline results"""
    if not results:
        print("[ERROR] No results to display")
        return
    
    print("\n" + "="*80)
    print("PYSPARK REDDIT SENTIMENT ANALYSIS - COMPLETE RESULTS")
    print("="*80)
    
    # Pipeline information
    pipeline_id = results.get("pipeline_id", "N/A")
    method = results.get("analysis_method", results.get("method", "N/A"))
    
    print(f"Pipeline ID: {pipeline_id}")
    print(f"Analysis Method: {method}")
    print(f"Timestamp: {results.get('analysis_timestamp', 'N/A')}")
    
    # Processing statistics
    total_comments = results.get("total_comments", 0)
    processing_time = results.get("processing_time_seconds", 0)
    
    print(f"\nProcessing Statistics:")
    print(f"  Total Comments: {total_comments}")
    print(f"  Processing Time: {processing_time:.2f} seconds")
    
    if total_comments > 0:
        avg_time = processing_time / total_comments
        print(f"  Average Time per Comment: {avg_time:.3f} seconds")
    
    # Sentiment distribution
    distribution = results.get("sentiment_distribution", {})
    if distribution:
        print(f"\nSentiment Distribution:")
        total_analyzed = sum(distribution.values())
        for sentiment, count in distribution.items():
            percentage = (count / total_analyzed) * 100 if total_analyzed > 0 else 0
            print(f"  {sentiment}: {count} comments ({percentage:.1f}%)")
        
        # Overall sentiment
        positive = distribution.get("Positive", 0)
        negative = distribution.get("Negative", 0)
        neutral = distribution.get("Neutral", 0)
        
        if positive > negative and positive > neutral:
            overall = "Positive"
        elif negative > positive and negative > neutral:
            overall = "Negative"
        else:
            overall = "Neutral"
            
        print(f"\nOverall Community Sentiment: {overall}")
    
    # Model information
    model = results.get("model_used", "N/A")
    print(f"\nModel Used: {model}")
    
    # Big data features (if applicable)
    if method == "pyspark" or "spark" in str(method).lower():
        print(f"\nBig Data Features:")
        print(f"  ✓ Distributed Processing")
        print(f"  ✓ Scalable Architecture") 
        print(f"  ✓ Advanced Analytics")
        print(f"  ✓ Real-time Capabilities")
    
    print("="*80)


def main():
    """Main pipeline execution"""
    print("PYSPARK REDDIT SENTIMENT ANALYSIS PIPELINE")
    print("="*80)
    print("Complete big data pipeline with distributed processing")
    print("="*80)
    
    start_time = time.time()
    
    try:
        # Step 1: Setup Spark environment
        print("\n[STEP 1] Setting up Spark environment...")
        if not setup_spark_environment():
            print("[ERROR] Spark environment setup failed")
            return False
        
        # Step 2: Test PySpark installation
        print("\n[STEP 2] Testing PySpark installation...")
        pyspark_available = test_pyspark_installation()
        
        # Step 3: Create directories
        print("\n[STEP 3] Setting up data directories...")
        create_data_directories()
        
        # Step 4: Fetch Reddit comments
        print("\n[STEP 4] Fetching Reddit comments...")
        if not run_reddit_fetch():
            print("[ERROR] Failed to fetch Reddit comments")
            return False
        
        # Step 5: Run sentiment analysis pipeline
        print("\n[STEP 5] Running sentiment analysis pipeline...")
        if pyspark_available:
            print("[INFO] Using PySpark distributed processing")
            results = run_pyspark_pipeline()
        else:
            print("[INFO] Using standard processing (PySpark not available)")
            results = run_standard_fallback()
        
        if not results:
            print("[ERROR] Pipeline execution failed")
            return False
        
        # Step 6: Display results
        print("\n[STEP 6] Displaying results...")
        display_comprehensive_results(results)
        
        # Calculate total time
        total_time = time.time() - start_time
        print(f"\n[SUCCESS] Pipeline completed in {total_time:.2f} seconds")
        print("[INFO] Results saved to output files")
        
        return True
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Pipeline stopped by user")
        return False
    except Exception as e:
        print(f"\n[FATAL] Pipeline failed: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    print(f"\n[EXIT] Pipeline {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)