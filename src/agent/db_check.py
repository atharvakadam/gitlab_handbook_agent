from pymongo import MongoClient
import os, collections
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
cli = MongoClient(MONGO_URI)
c = cli["gitlab_internal_documentation"]["handbook"]

# # Distinct/Counts for sections
# sections = c.distinct("section")
# print("sections:", len(sections))
# # Optional: top sections by count
# pipeline = [
#   {"$group": {"_id": "$section", "n": {"$sum": 1}}},
#   {"$sort": {"n": -1}},
#   {"$limit": 50}
# ]
# print(list(c.aggregate(pipeline)))

# # Distinct access groups
# print("access_groups:", c.distinct("access_groups"))

# # Distinct sources
# print("sources:", c.distinct("source"))

# # (tags are empty right now, will return [] or [""] depending)
# print("tags:", c.distinct("tags"))


import os, json
from pymongo import MongoClient
from filters import load_allowlists, suggest_sp1, build_vector_filter

client = MongoClient(os.environ["MONGO_URI"])
coll = client["gitlab_internal_documentation"]["handbook"]
allow = load_allowlists(coll)
print("ALLOW sizes:", {k: len(v) if hasattr(v,'__len__') else 'ok' for k,v in allow.items()})

print("suggest_sp1('parental leave in india'):", suggest_sp1("parental leave in india"))
print("filter example:",
      json.dumps(build_vector_filter(user_groups=["all"],
                                     sources=["gitlab-handbook"],
                                     sp1_any=["total-rewards"]),
                 default=str))