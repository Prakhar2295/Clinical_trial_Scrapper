import weaviate
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(port=8079, grpc_port=50060)
collection = client.collections.get("ClinicalTrialChunk")

# Step 1 — Find all distinct category values stored in DB
print("=== STEP 1: All distinct category values in Weaviate ===")
results = collection.query.fetch_objects(
    limit=10_000,
    return_properties=["category", "filename"]
)
categories = {}
for obj in results.objects:
    cat = obj.properties.get("category", "")
    fname = obj.properties.get("filename", "")
    if cat not in categories:
        categories[cat] = []
    if fname not in categories[cat]:
        categories[cat].append(fname)

for cat, files in sorted(categories.items()):
    print(f"  '{cat}' ({len(files)} files): {files}")

# Step 2 — Check if any category appears in multiple casings
print("\n=== STEP 2: Casing collision check ===")
lower_map = {}
for cat in categories:
    key = cat.lower()
    if key not in lower_map:
        lower_map[key] = []
    lower_map[key].append(cat)

collisions = {k: v for k, v in lower_map.items() if len(v) > 1}
if collisions:
    print("  ISSUE FOUND — same category stored in multiple casings:")
    for key, variants in collisions.items():
        print(f"  '{key}' appears as: {variants}")
else:
    print("  No casing collisions found — all categories consistent")

# Step 3 — Check what category_registry has
print("\n=== STEP 3: Category registry values ===")
import sys
sys.path.insert(0, '.')
from app.core.category_registry import category_registry
print("  Registry categories:", category_registry.all())

# Step 4 — Check if DB categories match registry exactly
print("\n=== STEP 4: DB vs Registry mismatch check ===")
db_cats = set(categories.keys())
reg_cats = set(category_registry.all())
in_db_not_registry = db_cats - reg_cats
in_registry_not_db = reg_cats - db_cats
if in_db_not_registry:
    print(f"  ISSUE: In DB but not in registry: {in_db_not_registry}")
else:
    print("  All DB categories exist in registry")
if in_registry_not_db:
    print(f"  INFO: In registry but not yet in DB: {in_registry_not_db}")

client.close()
