from ai.ingestion.ingestion_pipeline import IngestionPipeline

pipeline = IngestionPipeline()

doc = pipeline.ingest(
    "intellistudy/data/uploads/1-introduction to Aspect-Oriented Software Systems.pdf",
)

print(doc.statistics)
print(doc.full_text)