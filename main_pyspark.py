"""
PySpark Big Data Pipeline Orchestrator for Reddit Sentiment Analysis
Main orchestrator that coordinates the entire big data processing pipeline
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline modules
try:
    from spark_ingestion import RedditDataIngestion
    from spark_sentiment_processor import DistributedSentimentProcessor
    from spark_analytics import RedditSentimentAnalytics
except ImportError as e:
    print(f"Error importing pipeline modules: {e}")
    print("Make sure all Spark modules are in the same directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class RedditSentimentPipeline:
    """
    Main orchestrator for the Reddit Sentiment Analysis Big Data Pipeline.
    Coordinates ingestion, processing, sentiment analysis, and analytics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.pipeline_id = f"pipeline_{int(time.time())}"
        self.start_time = datetime.now()
        
        # Initialize components
        self.ingestion = None
        self.processor = None
        self.analytics = None
        
        # Pipeline state
        self.pipeline_stats = {
            "pipeline_id": self.pipeline_id,
            "start_time": self.start_time.isoformat(),
            "stages_completed": [],
            "errors": [],
            "processing_stats": {}
        }
        
        logger.info(f"Initialized pipeline: {self.pipeline_id}")
    
    def initialize_components(self):
        """Initialize all pipeline components."""
        logger.info("Initializing pipeline components")
        
        try:
            # Initialize ingestion
            self.ingestion = RedditDataIngestion(
                app_name=f"RedditIngestion_{self.pipeline_id}"
            )
            
            # Initialize sentiment processor
            self.processor = DistributedSentimentProcessor(
                app_name=f"SentimentProcessor_{self.pipeline_id}",
                batch_size=self.config.get("batch_size", 1000)
            )
            
            # Initialize analytics
            self.analytics = RedditSentimentAnalytics(
                app_name=f"SentimentAnalytics_{self.pipeline_id}"
            )
            
            logger.info("All components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            self.pipeline_stats["errors"].append(f"Initialization error: {str(e)}")
            raise
    
    def stage_1_data_ingestion(self) -> str:
        """Stage 1: Ingest and process Reddit comments."""
        logger.info("Starting Stage 1: Data Ingestion")
        stage_start = time.time()
        
        try:
            input_path = self.config["input"]["comments_file"]
            
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Ingest comments
            if self.config["input"].get("batch_mode", False):
                # Batch processing mode
                directory = os.path.dirname(input_path)
                df = self.ingestion.ingest_batch_files(directory, "*.json")
            else:
                # Single file mode
                df = self.ingestion.ingest_jsonl_comments(input_path)
            
            # Apply data quality filters
            clean_df = self.ingestion.filter_valid_comments(df)
            
            # Generate ingestion statistics
            stats = self.ingestion.get_ingestion_statistics(clean_df)
            self.pipeline_stats["processing_stats"]["ingestion"] = stats
            
            # Save processed data
            output_path = self.config["output"]["processed_data_path"]
            self.ingestion.save_processed_data(clean_df, output_path, "parquet")
            
            # Create partitioned dataset for optimal processing
            if self.config.get("create_partitions", True):
                partition_path = f"{output_path}_partitioned"
                self.ingestion.create_partitioned_dataset(clean_df, partition_path)
            
            stage_time = time.time() - stage_start
            self.pipeline_stats["stages_completed"].append({
                "stage": "data_ingestion",
                "duration_seconds": stage_time,
                "records_processed": stats["total_records"],
                "records_valid": stats["valid_records"]
            })
            
            logger.info(f"Stage 1 completed in {stage_time:.2f} seconds")
            return output_path
            
        except Exception as e:
            error_msg = f"Stage 1 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_stats["errors"].append(error_msg)
            raise
    
    def stage_2_sentiment_processing(self, input_path: str) -> str:
        """Stage 2: Perform distributed sentiment analysis."""
        logger.info("Starting Stage 2: Sentiment Processing")
        stage_start = time.time()
        
        try:
            # Load processed data
            df = self.processor.spark.read.parquet(input_path)
            
            # Preprocess text for sentiment analysis
            preprocessed_df = self.processor.preprocess_text_distributed(df)
            
            # Perform sentiment analysis
            sentiment_df = self.processor.process_sentiment_distributed(preprocessed_df)
            
            # Generate sentiment summary
            summary = self.processor.create_sentiment_summary(sentiment_df)
            self.pipeline_stats["processing_stats"]["sentiment_analysis"] = summary
            
            # Save sentiment results
            output_path = self.config["output"]["sentiment_results_path"]
            self.processor.save_sentiment_results(
                sentiment_df, 
                output_path, 
                include_original=self.config.get("include_original_text", True)
            )
            
            stage_time = time.time() - stage_start
            self.pipeline_stats["stages_completed"].append({
                "stage": "sentiment_processing",
                "duration_seconds": stage_time,
                "records_processed": summary["total_processed"]
            })
            
            logger.info(f"Stage 2 completed in {stage_time:.2f} seconds")
            return output_path
            
        except Exception as e:
            error_msg = f"Stage 2 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_stats["errors"].append(error_msg)
            raise
    
    def stage_3_analytics_processing(self, input_path: str) -> Dict[str, Any]:
        """Stage 3: Perform comprehensive analytics."""
        logger.info("Starting Stage 3: Analytics Processing")
        stage_start = time.time()
        
        try:
            # Load sentiment results
            df = self.analytics.load_sentiment_data(input_path)
            
            # Generate comprehensive analytics report
            report = self.analytics.generate_comprehensive_report(df)
            
            # Create dashboard data
            dashboard_data = self.analytics.create_sentiment_dashboard_data(df)
            
            # Save analytics results
            analytics_output = self.config["output"]["analytics_report_path"]
            dashboard_output = self.config["output"]["dashboard_data_path"]
            
            self.analytics.save_analytics_results(report, analytics_output)
            self.analytics.save_analytics_results(dashboard_data, dashboard_output)
            
            # Store analytics in pipeline stats
            self.pipeline_stats["processing_stats"]["analytics"] = {
                "report_path": analytics_output,
                "dashboard_path": dashboard_output,
                "total_insights_generated": len(report.keys())
            }
            
            stage_time = time.time() - stage_start
            self.pipeline_stats["stages_completed"].append({
                "stage": "analytics_processing",
                "duration_seconds": stage_time,
                "insights_generated": len(report.keys())
            })
            
            logger.info(f"Stage 3 completed in {stage_time:.2f} seconds")
            return report
            
        except Exception as e:
            error_msg = f"Stage 3 failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_stats["errors"].append(error_msg)
            raise
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete big data pipeline."""
        logger.info(f"Starting complete pipeline execution: {self.pipeline_id}")
        pipeline_start = time.time()
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Stage 1: Data Ingestion
            processed_data_path = self.stage_1_data_ingestion()
            
            # Stage 2: Sentiment Processing
            sentiment_results_path = self.stage_2_sentiment_processing(processed_data_path)
            
            # Stage 3: Analytics Processing
            analytics_report = self.stage_3_analytics_processing(sentiment_results_path)
            
            # Calculate total pipeline time
            total_time = time.time() - pipeline_start
            self.pipeline_stats["end_time"] = datetime.now().isoformat()
            self.pipeline_stats["total_duration_seconds"] = total_time
            self.pipeline_stats["status"] = "SUCCESS"
            
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            # Save pipeline statistics
            self.save_pipeline_stats()
            
            return {
                "status": "SUCCESS",
                "pipeline_id": self.pipeline_id,
                "duration": total_time,
                "analytics_report": analytics_report,
                "pipeline_stats": self.pipeline_stats
            }
            
        except Exception as e:
            total_time = time.time() - pipeline_start
            self.pipeline_stats["end_time"] = datetime.now().isoformat()
            self.pipeline_stats["total_duration_seconds"] = total_time
            self.pipeline_stats["status"] = "FAILED"
            
            error_msg = f"Pipeline failed: {str(e)}"
            logger.error(error_msg)
            self.pipeline_stats["errors"].append(error_msg)
            
            # Save pipeline statistics even on failure
            self.save_pipeline_stats()
            
            return {
                "status": "FAILED",
                "pipeline_id": self.pipeline_id,
                "duration": total_time,
                "error": str(e),
                "pipeline_stats": self.pipeline_stats
            }
        
        finally:
            self.cleanup_resources()
    
    def run_streaming_pipeline(self, checkpoint_dir: str):
        """Run streaming version of the pipeline for real-time processing."""
        logger.info("Starting streaming pipeline")
        
        try:
            self.initialize_components()
            
            # Set up streaming paths
            streaming_input = self.config["streaming"]["input_path"]
            streaming_output = self.config["streaming"]["output_path"]
            
            # Start streaming sentiment processing
            query = self.processor.process_streaming_sentiment(
                streaming_input, streaming_output, checkpoint_dir
            )
            
            logger.info("Streaming pipeline started, waiting for termination...")
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Streaming pipeline failed: {str(e)}")
            raise
        finally:
            self.cleanup_resources()
    
    def save_pipeline_stats(self):
        """Save pipeline execution statistics."""
        stats_file = self.config["output"].get("pipeline_stats_path", 
                                                f"data/pipeline_stats_{self.pipeline_id}.json")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(stats_file), exist_ok=True)
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.pipeline_stats, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Pipeline statistics saved to: {stats_file}")
    
    def cleanup_resources(self):
        """Cleanup Spark sessions and resources."""
        logger.info("Cleaning up pipeline resources")
        
        try:
            if self.ingestion:
                self.ingestion.stop_session()
            if self.processor:
                self.processor.stop_session()
            if self.analytics:
                self.analytics.stop_session()
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
        
        logger.info("Resource cleanup completed")


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """Load pipeline configuration from file."""
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        # Return default configuration
        return get_default_config()


def get_default_config() -> Dict[str, Any]:
    """Get default pipeline configuration."""
    return {
        "input": {
            "comments_file": "comments.json",
            "batch_mode": False
        },
        "output": {
            "processed_data_path": "data/processed_comments",
            "sentiment_results_path": "data/sentiment_results",
            "analytics_report_path": "data/analytics_report.json",
            "dashboard_data_path": "data/dashboard_data.json",
            "pipeline_stats_path": "data/pipeline_stats.json"
        },
        "processing": {
            "batch_size": 1000,
            "create_partitions": True,
            "include_original_text": True
        },
        "streaming": {
            "input_path": "streaming_input/",
            "output_path": "streaming_output/",
            "checkpoint_path": "streaming_checkpoints/"
        }
    }


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Reddit Sentiment Analysis Big Data Pipeline")
    parser.add_argument("--config", default="pipeline_config.json", 
                       help="Path to pipeline configuration file")
    parser.add_argument("--mode", choices=["batch", "streaming"], default="batch",
                       help="Pipeline execution mode")
    parser.add_argument("--input", help="Override input file path")
    parser.add_argument("--output-dir", help="Override output directory")
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_pipeline_config(args.config)
        
        # Apply command line overrides
        if args.input:
            config["input"]["comments_file"] = args.input
        if args.output_dir:
            for key in config["output"]:
                config["output"][key] = os.path.join(args.output_dir, 
                                                   os.path.basename(config["output"][key]))
        
        # Create pipeline
        pipeline = RedditSentimentPipeline(config)
        
        print(f"\\n=== Reddit Sentiment Analysis Big Data Pipeline ===")
        print(f"Pipeline ID: {pipeline.pipeline_id}")
        print(f"Mode: {args.mode}")
        print(f"Input: {config['input']['comments_file']}")
        print(f"Config: {args.config}")
        print("=" * 50)
        
        # Run pipeline based on mode
        if args.mode == "batch":
            result = pipeline.run_complete_pipeline()
            
            print(f"\\n=== Pipeline Results ===")
            print(f"Status: {result['status']}")
            print(f"Duration: {result['duration']:.2f} seconds")
            print(f"Pipeline ID: {result['pipeline_id']}")
            
            if result['status'] == 'SUCCESS':
                print(f"\\n✅ Pipeline completed successfully!")
                print(f"📊 Analytics report: {config['output']['analytics_report_path']}")
                print(f"📈 Dashboard data: {config['output']['dashboard_data_path']}")
                
                # Print key insights
                if 'analytics_report' in result:
                    analytics = result['analytics_report']
                    if 'basic_statistics' in analytics:
                        stats = analytics['basic_statistics']
                        print(f"\\n📋 Key Insights:")
                        print(f"   • Total comments processed: {stats.get('total_records', 0):,}")
                        if 'sentiment_distribution' in stats:
                            for sentiment, data in stats['sentiment_distribution'].items():
                                print(f"   • {sentiment.title()} sentiment: {data['count']:,} ({data['percentage']}%)")
            else:
                print(f"\\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
        
        else:  # streaming mode
            checkpoint_dir = config["streaming"]["checkpoint_path"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            pipeline.run_streaming_pipeline(checkpoint_dir)
    
    except KeyboardInterrupt:
        print("\\n🛑 Pipeline interrupted by user")
    except Exception as e:
        print(f"\\n💥 Pipeline failed with error: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()