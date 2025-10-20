from pymongo import MongoClient
import os, collections
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
cli = MongoClient(MONGO_URI)
c = cli["gitlab_internal_documentation"]["handbook"]

# Distinct/Counts for sections
sections = c.distinct("section")
print("sections:", len(sections))
# Optional: top sections by count
pipeline = [
  {"$group": {"_id": "$section", "n": {"$sum": 1}}},
  {"$sort": {"n": -1}},
  {"$limit": 50}
]
print(list(c.aggregate(pipeline)))

# Distinct access groups
print("access_groups:", c.distinct("access_groups"))

# Distinct sources
print("sources:", c.distinct("source"))

# (tags are empty right now, will return [] or [""] depending)
print("tags:", c.distinct("tags"))


