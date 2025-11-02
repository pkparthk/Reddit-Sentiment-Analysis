import subprocess
import sys
import os
import json
import time
from datetime import datetime


def run_reddit_fetch():
    """Fetch fresh Reddit comments using Node.js script"""
    print("📡 Fetching Reddit comments...")
    try:        
        result = subprocess.run(
            "node fetch.js", 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd(),
            encoding='utf-8',
            errors='ignore'
        )
        if result.returncode == 0:
            print("✅ Reddit comments fetched successfully")
            return True
        else:
            print(f"❌ Failed to fetch comments: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error fetching comments: {str(e)}")
        return False


def check_pyspark_available():
    """Check if PySpark is properly configured"""
    try:
        from pyspark.sql import SparkSession
        return True
    except Exception as e:
        print(f"⚠️  PySpark not available: {str(e)}")
        return False


def load_pipeline_config():
    """Load pipeline configuration from JSON file"""
    try:
        with open('pipeline_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load pipeline config: {e}")        
        return {
            "input": {"comments_file": "comments.json"},
            "processing": {"batch_size": 32},
            "sentiment": {"model_name": "barissayil/bert-sentiment-analysis-sst"}
        }


def run_pyspark_analysis():    
    print("🚀 Starting PySpark sentiment analysis pipeline...")
    
    if check_pyspark_available():
        try:            
            config = load_pipeline_config()
                                
            from simple_integrated_pipeline import SimplifiedIntegratedPipeline
            
            pipeline = SimplifiedIntegratedPipeline(config)
                        
            start_time = time.time()
            success = pipeline.run_pipeline()
            execution_time = time.time() - start_time
            
            if success:                
                analytics_file = os.path.join("output", "analytics_summary.json")
                if os.path.exists(analytics_file):
                    with open(analytics_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                                        
                    results["execution_time"] = execution_time
                else:                    
                    results = {
                        "pipeline_metadata": {
                            "pipeline_id": getattr(pipeline, 'pipeline_id', 'unknown'),
                            "total_comments": 0,
                            "processing_method": "PySpark"
                        },
                        "execution_time": execution_time
                    }
            
            if results and results.get('pipeline_metadata', {}).get('total_comments', 0) > 0:
                print("✅ PySpark analysis completed successfully")
                return results
            else:
                print("⚠️ PySpark pipeline returned no results")
                return run_standard_analysis()
            
        except Exception as e:
            print(f"❌ PySpark pipeline error: {str(e)}")
            print("🔄 Falling back to standard analysis...")
            return run_standard_analysis()
    else:
        print("🔄 Running standard sentiment analysis...")
        return run_standard_analysis()


def run_standard_analysis():
    """Run standard sentiment analysis without PySpark"""
    print("🧠 Analyzing sentiment with trained model...")
    
    try:        
        from arguments import args
        from analyzer import Analyzer
                
        if not os.path.exists("comments.json"):
            print("❌ No comments file found")
            return None
            
        comments = []
        with open("comments.json", 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    comment_data = json.loads(line)
                    comments.append(comment_data['text'])
        
        print(f"📊 Processing {len(comments)} comments...")
                
        analyzer = Analyzer(will_train=False, args=args)
                
        results = []
        sentiment_counts = {"Positive": 0, "Negative": 0}
        start_time = time.time()
        
        for i, comment in enumerate(comments, 1):
            sentiment, confidence = analyzer.classify_sentiment(comment)
            
            results.append({
                "comment_id": i,
                "comment": comment,
                "sentiment": sentiment,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            })
            
            sentiment_counts[sentiment] += 1
                        
            if i % 5 == 0 or i == len(comments):
                print(f"  Processed {i}/{len(comments)} comments...")
        
        processing_time = time.time() - start_time
                
        output_data = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_comments": len(comments),
            "processing_time_seconds": processing_time,
            "sentiment_distribution": sentiment_counts,
            "results": results
        }
        
        with open("reddit_sentiment_results.json", 'w', encoding='utf-8') as file:
            json.dump(output_data, file, indent=2, ensure_ascii=False)
        
        return output_data
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return None


def display_results(results):
    """Display analysis results"""
    if not results:
        print("\n❌ No results to display")
        return
        
    print("\n" + "="*60)
    print("📈 REDDIT SENTIMENT ANALYSIS RESULTS")
    print("="*60)
        
    if "pipeline_metadata" in results:        
        metadata = results.get("pipeline_metadata", {})
        total = metadata.get("total_comments", 0)
        processing_time = results.get("execution_time", 0)
        method = metadata.get("processing_method", "Unknown")
        pipeline_id = metadata.get("pipeline_id", "N/A")
                
        distribution = results.get("sentiment_distribution", {})
        confidence_avg = results.get("average_confidence", {})
        
        print(f"Pipeline ID: {pipeline_id}")
        print(f"Processing Method: {method}")
        print(f"Total Comments Analyzed: {total}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Average Time per Comment: {processing_time/total:.3f} seconds" if total > 0 else "")
        
        print("\n🎯 SENTIMENT DISTRIBUTION:")
        for sentiment, count in distribution.items():
            percentage = (count / total) * 100 if total > 0 else 0
            avg_conf = confidence_avg.get(sentiment, 0)
            print(f"  {sentiment}: {count} comments ({percentage:.1f}%) - Avg Confidence: {avg_conf:.1f}%")
            
    else:        
        total = results.get("total_comments", 0)
        processing_time = results.get("processing_time_seconds", 0)
        distribution = results.get("sentiment_distribution", {})
        
        print(f"Processing Method: Standard Analysis")
        print(f"Total Comments Analyzed: {total}")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print(f"Average Time per Comment: {processing_time/total:.3f} seconds" if total > 0 else "")
        
        print("\n🎯 SENTIMENT DISTRIBUTION:")
        for sentiment, count in distribution.items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {sentiment}: {count} comments ({percentage:.1f}%)")
        
    positive = distribution.get("Positive", 0)
    negative = distribution.get("Negative", 0)
    
    if positive > negative:
        overall = "Positive"
    elif negative > positive:
        overall = "Negative"
    else:
        overall = "Neutral"
    
    print(f"\n🏆 Overall Community Sentiment: {overall}")
        
    output_files = []
    if os.path.exists("reddit_sentiment_results.json"):
        output_files.append("reddit_sentiment_results.json")
    if os.path.exists("output"):
        output_files.append("output/ directory")
    
    if output_files:
        print(f"📁 Results saved to: {', '.join(output_files)}")
    
    print("\n✨ Analysis complete!")


def main():
    """Main execution function"""
    print("🚀 REDDIT SENTIMENT ANALYSIS WITH BIG DATA PROCESSING")    
        
    if not run_reddit_fetch():
        print("❌ Pipeline aborted - could not fetch Reddit comments")
        return
        
    results = run_pyspark_analysis()
        
    display_results(results)
    
    print("\n🎉 Analysis pipeline completed!")
    print("🔄 Run 'python main.py' again for fresh analysis")


if __name__ == "__main__":
    main()