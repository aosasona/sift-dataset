# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "openai",
#     "python-dotenv",
#     "readability-python",
#     "requests",
# ]
# ///
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from dotenv import load_dotenv
from openai import OpenAI
from readability import Readability

load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
}

TAG_CATEGORIES = {
    "ai": ["ai"],
    "computer science": ["compsci", "formalmethods", "networking", "plt"],
    "culture": ["culture", "person", "philosophy"],
    "law": ["law"],
    "cognitive science": ["cogsci"],
    "cryptography": ["cryptography"],
    "education": ["education"],
    "finance": ["finance"],
    "hardware": ["hardware"],
    "mathematics": ["math"],
    "science": ["science"],
    "art": ["art"],
    "retrospective": ["historical", "retrocomputing"],
    "rant": ["rant"],
    "satire": ["satire"],
    "design": ["a11y", "design", "visualization"],
    "updates": ["release", "email"],
    "operating systems": ["osdev", "android", "dragonflybsd", "freebsd", "illumos", "ios", "linux", "mac", "netbsd", "nix", "openbsd", "unix", "windows"],
    "games": ["games"],
    "cybersecurity": ["reversing", "security", "privacy"],
    "devops": ["devops", "performance", "scaling", "testing", "virtualization", "distributed"],
    "programming": [
        "graphics", "databases", "merkle-trees", "mobile", "editors", "emacs", "systemd", "vcs", "vim", "vscode",
        "programming", "compilers", "browsers", "ipv6", "wasm", "web", "api", "debugging", "practices", "apl",
        "assembly", "c", "c++", "clojure", "css", "d", "dotnet", "elixir", "elm", "erlang", "fortran", "gleam", "go",
        "haskell", "java", "javascript", "kotlin", "lisp", "lua", "ml", "nodejs", "objectivec", "perl", "php",
        "python", "ruby", "rust", "scala", "swift", "zig"
    ],
}
TAG_TO_CATEGORY = {tag: cat for cat, tags in TAG_CATEGORIES.items() for tag in tags}

MIN_CATEGORY_COUNT = 100
MAX_PAGES = 25
# Structure : { tag: <min-count> / <category tags count> }
CATEGORY_TAGS_COUNT = {tag: max(MIN_CATEGORY_COUNT // len(tags), 4) for tag, tags in TAG_CATEGORIES.items()}
TAGS = list(TAG_TO_CATEGORY.keys())

EXAMPLE_SUMMARIES = [
    {
        "title": "Over 4 Million Americans Roll Up Sleeves For Omicron-Targeted COVID Boosters",
        "summary": "Health experts said it is too early to predict whether demand would match up with the 171 million doses of the new boosters the U.S. ordered for the fall."
    },
    {
        "title": "Twitch Bans Gambling Sites After Streamer Scams Folks Out Of $200,000",
        "summary": "One man's claims that he scammed people on the platform caused several popular streamers to consider a Twitch boycott.",
    }
]

POSTS = {}

IGNORE_LINKS = [
    "youtube.com",
    "vimeo.com",
    "github.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "reddit.com",
    "news.ycombinator.com",
    "bsky.app",
    "bsky.com",
    "t.co",
    "t.me",
    "twitch.tv",
    "tiktok.com",
    "instagram.com",
    "facebook.com",
    "linkedin.com",
    "linkedin.co.uk",
    "pinterest.com",
]
IGNORE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
    ".wmv",
    ".flv",
    ".mkv",
    ".pdf"
]

parser = Readability()
lock = Lock()


def thread_safe_add_post(id, post):
    with lock:
        if id not in POSTS:
            POSTS[id] = post
            return True
        else:
            return False


def fetch_content(url: str) -> object | None:
    html_content = ""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        response.raise_for_status()  # Raise an error for bad responses

        # Ensure it is an HTML page
        if not response.headers.get("Content-Type", "").startswith("text/html"):
            print(f"Non-HTML content for URL: {url}")
            return None

        html_content = response.text
        if len(html_content) == 0:
            print(f"Empty content for URL: {url}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}, Code: {e.response.status_code if e.response else 'N/A'}")
        return None

    article, err = parser.parse(html_content, url=url)
    if err:
        print(f"Error parsing content {url}: {err}")
        return None

    return article


def get_url(tag: str, page: int) -> str:
    """
    Get the URL for a specific tag and page number.
    """
    return f"https://lobste.rs/t/{tag}.json?page={page}"


def generate_summary(title: str, content: str) -> str:
    prompt = "Based on the examples below, write a short news-style summary/excerpt for the given article, limited to two or three lines at most; about 40-60 words.\n\n"
    for example in EXAMPLE_SUMMARIES:
        prompt += f"Title: {example['title']}\nSummary: {example['summary']}\n\n"

    prompt += f"Title: {title}\nContent: {content}\nSummary: "

    completion = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes tech news articles in a concise style."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        n=1
    )

    return completion.choices[0].message.content.strip()


def fetch_tag(tag: str) -> None:
    """
    Fetch posts for a specific tag from Lobsters using the min count.

    Before adding a post to the list, check if the post already exists in the list.
    """

    items_count = 0
    page = 1
    category = TAG_TO_CATEGORY.get(tag, "other")
    MIN_COUNT = CATEGORY_TAGS_COUNT.get(category, MIN_CATEGORY_COUNT)
    STATS = {"tag": tag, "category": category, "min_count": MIN_COUNT}
    print(json.dumps(STATS, indent=4))

    while True:
        try:
            # When we have the minimum amount of posts, we can stop fetching pages.
            if items_count >= MIN_COUNT or page > MAX_PAGES:
                break

            url = get_url(tag, page)
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
            if response.status_code != 200:
                print(f"fetch_tag: Error fetching URL: {url}, Code: {response.status_code}")
                break

            data = response.json()

            if len(data) == 0:
                break

            for post in data:
                if items_count >= MIN_COUNT:
                    return

                id = post['short_id']

                for ignore in IGNORE_LINKS:
                    if ignore in post["url"]:
                        continue

                for ext in IGNORE_EXTENSIONS:
                    if post["url"].endswith(ext):
                        continue

                if id in POSTS:
                    continue

                parsed = fetch_content(post["url"])
                if parsed is None or parsed.text_content.strip() == "":
                    continue

                summary = generate_summary(post["title"], parsed.text_content)
                if summary is None or summary.strip() == "":
                    continue

                post = {
                    "link": post["url"],
                    "headline": post["title"],
                    "category": TAG_TO_CATEGORY.get(tag, "other").upper(),
                    "short_description": summary,
                    "date": post["created_at"],
                }
                if thread_safe_add_post(id, post):
                    items_count += 1

            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching tag {tag} on page {page}: {e}")
            break
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON for tag {tag} on page {page}: {e}")
            break
        except Exception as e:
            print(f"Unexpected error for tag {tag} on page {page}: {e}")
            break


def write_to_json(posts: dict, filename: str) -> None:
    if not filename.endswith(".json"):
        filename += ".json"

    arr = list(posts.values())
    content = json.dumps(arr, indent=4, ensure_ascii=False)
    with open(filename, "w") as f:
        f.write(content)


def fetch_all_tags() -> None:
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_tag, tag) for tag in TAGS]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error fetching tag: {e}")


def main():
    # fetch_all_tags()
    for tag in TAGS:
        print(f"Fetching posts for tag: {tag}")
        fetch_tag(tag)
        print(f"Finished fetching posts for tag: {tag}")

    print(">> Finished fetching posts for all tags.")
    print(f">> Total posts fetched: {len(POSTS)}")

    write_to_json(POSTS, "lobsters_dataset.json")


if __name__ == "__main__":
    main()
