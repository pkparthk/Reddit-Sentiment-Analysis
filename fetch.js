import axios from "axios";
import dotenv from "dotenv";
import fs from "fs";
import { json } from "stream/consumers";
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
  const score =
    typeof commentData.score === "number" ? commentData.score : null;
  const depth =
    typeof commentData.depth === "number" ? commentData.depth : null;

  outComments.push({ id, author, body, score, depth });

  const replies = commentData.replies;
  if (isObject(replies)) {
    collectFromListing(replies, outComments);
  }
}

function main(data) {
  const jsonText = data;
  const root = JSON.parse(jsonText);

  const comments = [];
  collectFromListing(root, comments);

  const jsonl = comments
    .map((c) => JSON.stringify({ text: c.body }))
    .join("\n");
  fs.writeFileSync("comments.json", jsonl + "\n", "utf8");
  fs.writeFileSync(
    "comments.txt",
    comments.map((c) => c.body).join("\n\n---\n\n"),
    "utf8"
  );

  console.log(
    `[SUCCESS] Extracted ${comments.length} comments -> comments.json (JSONL), comments.txt`
  );
}
const fetch_data = async (url) => {
  const u = new URL(url);
  const parts = u.pathname.split("/").filter(Boolean);
  const subreddit = parts[1];
  const commentCode = parts[3];
  const apiUrl = `https://oauth.reddit.com/r/${subreddit}/comments/${commentCode}`;

  const data = await axios.get(apiUrl, {
    headers: {
      "User-Agent": `${process.env.reddit_username}`,
      Authorization: `bearer ${process.env.token}`,
    },
  });
  const Comment_data = JSON.stringify(data.data[1], null, 2);
  main(Comment_data);
};
const url =
  "https://www.reddit.com/r/IndianTeenagers/comments/1on5iut/guys_pls_save_me/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button";
fetch_data(url);
