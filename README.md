# Reddit Sentiment Analysis

A comprehensive sentiment analysis pipeline for Reddit comments with both standard processing and distributed PySpark capabilities. The system fetches Reddit comments and analyzes sentiment using BERT-based models or rule-based fallbacks.

## 🚀 Features

- **Multi-Modal Processing**: Standard pandas processing or distributed PySpark for big data
- **Reddit Integration**: Automated comment fetching via [fetch.js](fetch.js)
- **Advanced Sentiment Analysis**: BERT-based models with confidence scoring
- **Comprehensive Analytics**: Statistical analysis, confidence metrics, and temporal patterns
- **Flexible Configuration**: JSON-based pipeline configuration
- **Fallback Processing**: Graceful degradation when dependencies unavailable

## 🏃‍♂️ Quick Start

### 1. Setup Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Reddit Access
Create a `.env` file with Reddit API credentials:
```env
reddit_username=your_username
token=your_oauth_token
```

### 3. Run Analysis

#### Standard Processing (Recommended for beginners)
```bash
python main.py
```

#### PySpark Distributed Processing
```bash
# Full PySpark pipeline
python run_pyspark_pipeline.py

# Or use the orchestrator
python main_pyspark.py --config pipeline_config.json --mode batch
```

#### Simplified Pipeline
```bash
python simple_integrated_pipeline.py --batch
```

## 🔧 Pipeline Components

### Data Fetching
The [`fetch.js`](fetch.js) script extracts comments from Reddit posts:
```bash
node fetch.js "<reddit_post_url>"
```
Outputs: `comments.json` (JSONL format) and `comments.txt`

### Sentiment Analysis
Core analysis handled by [`Analyzer`](analyzer.py) class:
- BERT-based sentiment classification
- Confidence scoring (0-100%)
- Positive/Negative sentiment detection

### Distributed Processing
PySpark components for large-scale analysis:
- [`RedditDataIngestion`](spark_ingestion.py) - Data loading and validation
- [`DistributedSentimentProcessor`](spark_sentiment_processor.py) - Scalable sentiment analysis
- [`RedditSentimentAnalytics`](spark_analytics.py) - Advanced statistical analysis

## 📊 Output Files

- `reddit_sentiment_results.json` - Complete analysis results
- `output/detailed_results.json` - Detailed per-comment results
- `output/analytics_summary.json` - Statistical summary
- `output/results.csv` - CSV format results
- `data/processed_comments/` - Preprocessed data (Parquet format)

## 🧪 Testing

Run the test suite:
```bash
pytest -q
```

Test files in [`tests/`](tests/) directory:
- [`test_analyzer.py`](tests/test_analyzer.py) - Test sentiment analysis
- [`test_arguments.py`](tests/test_arguments.py) - Test CLI arguments
- [`test_evaluate.py`](tests/test_evaluate.py) - Test model evaluation

## ⚙️ Configuration

### Pipeline Configuration
Edit [`pipeline_config.json`](pipeline_config.json) to customize:
- Spark settings (memory, cores)
- Model parameters
- Input/output paths
- Processing options

### Model Arguments
Configure via [`arguments.py`](arguments.py):
- Model path: `model_name_or_path`
- Batch sizes: `batch_size`
- Sequence lengths: `maxlen_train`, `maxlen_val`

## 🔍 Advanced Features

### Custom Model Training
```bash
python train.py --model_name_or_path bert-base-uncased --num_eps 3
```

### Model Evaluation
```bash
python evaluate.py
```

### Streaming Processing (PySpark)
```bash
python main_pyspark.py --mode streaming
```

## 🛠 Troubleshooting

### Common Issues

1. **PySpark Installation**:
   - Ensure Java 8/11 is installed
   - Check `JAVA_HOME` environment variable
   - Run [`run_pyspark_pipeline.py`](run_pyspark_pipeline.py) for environment setup

2. **Reddit Fetching Fails**:
   - Verify `.env` file contains valid credentials
   - Check Reddit API rate limits
   - Ensure Node.js is installed for [`fetch.js`](fetch.js)

3. **Model Loading Issues**:
   - Pipeline automatically falls back to simple keyword-based analysis
   - Check internet connection for model downloads
   - Verify model path in configuration

### Fallback Processing
The system gracefully handles missing dependencies:
- No PySpark → pandas processing
- No transformers → keyword-based sentiment
- No Reddit data → uses existing `comments.json`
