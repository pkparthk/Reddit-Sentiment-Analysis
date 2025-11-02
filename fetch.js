import axios from "axios";
import dotenv from "dotenv";
import fs from "fs";

dotenv.config();

function isObject(value) {
  return value !== null && typeof value === "object";
}

function collectFromListing(listing, outComments) {
  if (
    !isObject(listing) ||
    !isObject(listing.data) ||
    !Array.isArray(listing.data.children)
  )
    return;

  for (const child of listing.data.children) {
    if (!isObject(child) || !isObject(child.data)) continue;
    if (child.kind === "t1") {
      processComment(child.data, outComments);
    }
  }
}

function processComment(commentData, outComments) {
  const body = typeof commentData.body === "string" ? commentData.body : "";
  const author =
    typeof commentData.author === "string" ? commentData.author : "";
  const id = typeof commentData.id === "string" ? commentData.id : "";
  const score = typeof commentData.score === "number" ? commentData.score : 0;
  const created_utc =
    typeof commentData.created_utc === "number"
      ? commentData.created_utc
      : Date.now() / 1000;

  // Only add non-empty comments
  if (body.trim() && body !== "[deleted]" && body !== "[removed]") {
    outComments.push({ id, author, body, score, created_utc });
  }

  // Process nested replies
  const replies = commentData.replies;
  if (isObject(replies)) {
    collectFromListing(replies, outComments);
  }
}

function processCommentsForSpark(comments, options = {}) {
  const timestamp = new Date().toISOString();

  // Enhanced comment structure optimized for Spark processing
  const sparkComments = comments.map((c, index) => ({
    id: c.id || `comment_${index}`,
    author: c.author || "unknown",
    text: c.body || "",
    score: c.score || 0,
    created_utc: c.created_utc,
    subreddit: options.subreddit || "unknown",
    post_id: options.postId || "unknown",
    extraction_timestamp: timestamp,
    comment_length: (c.body || "").length,
    word_count: (c.body || "").split(/\s+/).filter((w) => w.length > 0).length,
    contains_url: /https?:\/\//.test(c.body || ""),
    is_question: /\?/.test(c.body || ""),
    sentiment_label: null, // To be filled by PySpark pipeline
    confidence_score: null, // To be filled by PySpark pipeline
  }));

  // Output in JSONL format for Spark
  const jsonl = sparkComments.map((c) => JSON.stringify(c)).join("\n");

  // Save to file for PySpark ingestion
  const outputFile = options.outputFile || "reddit_comments_spark.jsonl";
  fs.writeFileSync(outputFile, jsonl + "\n", "utf8");

  console.log(
    `[SUCCESS] Extracted ${sparkComments.length} comments -> ${outputFile} (JSONL for Spark)`
  );

  // Also save metadata for pipeline
  const metadata = {
    extraction_time: timestamp,
    total_comments: sparkComments.length,
    subreddit: options.subreddit,
    post_id: options.postId,
    source_url: options.sourceUrl,
    processing_ready: true,
  };

  fs.writeFileSync(
    "reddit_metadata.json",
    JSON.stringify(metadata, null, 2),
    "utf8"
  );

  return sparkComments;
}

const fetchRedditComments = async (url, options = {}) => {
  try {
    console.log(`[FETCH] Processing Reddit URL: ${url}`);

    const u = new URL(url);
    const parts = u.pathname.split("/").filter(Boolean);
    const subreddit = parts[1];
    const commentCode = parts[3];
    const apiUrl = `https://oauth.reddit.com/r/${subreddit}/comments/${commentCode}`;

    console.log(`[FETCH] Getting comments from: ${subreddit}/${commentCode}`);

    const response = await axios.get(apiUrl, {
      headers: {
        "User-Agent": `${process.env.reddit_username}`,
        Authorization: `bearer ${process.env.token}`,
      },
    });

    const commentData = JSON.stringify(response.data[1], null, 2);
    const root = JSON.parse(commentData);

    const comments = [];
    collectFromListing(root, comments);

    // Process for Spark integration
    const processOptions = {
      ...options,
      subreddit: subreddit,
      postId: commentCode,
      sourceUrl: url,
      outputFile: options.outputFile || "reddit_comments_spark.jsonl",
    };

    return processCommentsForSpark(comments, processOptions);
  } catch (error) {
    console.error(`[ERROR] Failed to fetch Reddit data: ${error.message}`);
    throw error;
  }
};

// Load URLs from configuration
function loadRedditUrls() {
  try {
    if (fs.existsSync("reddit_urls.txt")) {
      const content = fs.readFileSync("reddit_urls.txt", "utf8");
      const urls = content
        .split("\n")
        .map((line) => line.trim())
        .filter(
          (line) => line && !line.startsWith("#") && line.startsWith("http")
        );

      console.log(
        `[CONFIG] Loaded ${urls.length} Reddit URLs from config file`
      );
      return urls;
    }
  } catch (error) {
    console.warn(`[CONFIG] Error loading reddit_urls.txt: ${error.message}`);
  }

  // Fallback URL
  return [
    "https://www.reddit.com/r/IndianWorkplace/comments/1okm5yv/got_a_reality_check_in_the_interview/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button",
  ];
}

// Batch processing for multiple Reddit posts
const fetchMultiplePosts = async (urls, options = {}) => {
  console.log(
    `[BATCH] Processing ${urls.length} Reddit posts for Spark ingestion`
  );

  const allComments = [];
  const batchMetadata = {
    batch_id: `batch_${Date.now()}`,
    start_time: new Date().toISOString(),
    urls: urls,
    total_posts: urls.length,
  };

  for (let i = 0; i < urls.length; i++) {
    try {
      console.log(`[BATCH] Processing ${i + 1}/${urls.length}: ${urls[i]}`);

      const comments = await fetchRedditComments(urls[i], {
        outputFile: `reddit_comments_batch_${i}.jsonl`,
        batchIndex: i,
      });

      allComments.push(...comments);

      // Rate limiting
      if (i < urls.length - 1) {
        await new Promise((resolve) =>
          setTimeout(resolve, options.delayMs || 1000)
        );
      }
    } catch (error) {
      console.error(
        `[BATCH ERROR] Failed to process ${urls[i]}: ${error.message}`
      );
    }
  }

  // Combine all comments for Spark processing
  const combinedJsonl = allComments.map((c) => JSON.stringify(c)).join("\n");
  fs.writeFileSync(
    "reddit_comments_combined.jsonl",
    combinedJsonl + "\n",
    "utf8"
  );

  batchMetadata.end_time = new Date().toISOString();
  batchMetadata.total_comments = allComments.length;
  batchMetadata.success_rate = allComments.length > 0 ? "100%" : "0%";

  fs.writeFileSync(
    "batch_metadata.json",
    JSON.stringify(batchMetadata, null, 2),
    "utf8"
  );

  console.log(
    `[BATCH SUCCESS] Processed ${allComments.length} total comments -> reddit_comments_combined.jsonl`
  );
  return allComments;
};

// Main execution function
async function main() {
  console.log("🚀 Reddit Comment Fetcher for PySpark Pipeline");
  console.log("=".repeat(50));

  try {
    const urls = loadRedditUrls();

    if (urls.length === 0) {
      throw new Error("No valid Reddit URLs found");
    }

    // Check for batch mode
    const args = process.argv.slice(2);
    const isBatch = args.includes("--batch");

    if (isBatch && urls.length > 1) {
      console.log("[MODE] Batch processing multiple posts");
      await fetchMultiplePosts(urls, { delayMs: 2000 });
    } else {
      console.log("[MODE] Single post processing");
      await fetchRedditComments(urls[0]);
    }

    console.log("✅ Reddit data fetching completed successfully!");
    console.log("📁 Files ready for PySpark ingestion:");
    console.log("   - reddit_comments_spark.jsonl (or combined file)");
    console.log("   - reddit_metadata.json");
  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    process.exit(1);
  }
}

// Export for module usage
export { fetchRedditComments, fetchMultiplePosts, loadRedditUrls };

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}
