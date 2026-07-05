import weaviate
from weaviate.classes.query import Filter

client = weaviate.connect_to_local(port=8079, grpc_port=50060)
collection = client.collections.get("ClinicalTrialChunk")

# Find all chunks with non-canonical category casings
results = collection.query.fetch_objects(
    limit=10_000,
    return_properties=["filename", "category"]
)

CANONICAL = {
    "protocol": "Protocol",
    "sap": "SAP",
    "icf": "ICF",
    "csr": "CSR",
    "ib": "IB",
    "combined": "Combined",
}

fixed = 0
for obj in results.objects:
    cat = obj.properties.get("category", "")
    canonical = CANONICAL.get(cat.lower())
    if canonical and cat != canonical:
        collection.data.update(
            uuid=obj.uuid,
            properties={"category": canonical}
        )
        print(f"Fixed: {obj.properties.get('filename')} "
              f"'{cat}' → '{canonical}'")
        fixed += 1

print(f"\nTotal fixed: {fixed} chunks")
client.close()
