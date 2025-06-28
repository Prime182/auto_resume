import os
from pymongo import MongoClient
from bson.objectid import ObjectId

# Database connection details
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DATABASE_NAME = "ai_resume"
COLLECTION_NAME = "resume_templates"

def store_template_in_db():
    try:
        client = MongoClient(MONGO_URI)
        db = client[DATABASE_NAME]
        collection = db[COLLECTION_NAME]

        # Paths to template files
        base_path = "templates"
        html_path = os.path.join(base_path, "resume_template.html")
        css_path = os.path.join(base_path, "style.css")
        js_path = os.path.join(base_path, "script.js")

        # Read file contents
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        with open(js_path, "r", encoding="utf-8") as f:
            js_content = f.read()

        # Prepare document to insert
        template_data = {
            "template_id": "default_resume_template", # Using a fixed ID for now, can be dynamic
            "html_content": html_content,
            "css_content": css_content,
            "js_content": js_content
        }

        # Check if template_id already exists and update, otherwise insert
        result = collection.update_one(
            {"template_id": template_data["template_id"]},
            {"$set": template_data},
            upsert=True
        )

        if result.upserted_id:
            print(f"Template 'default_resume_template' inserted with _id: {result.upserted_id}")
        elif result.modified_count > 0:
            print(f"Template 'default_resume_template' updated successfully.")
        else:
            print(f"Template 'default_resume_template' already exists and no changes were made.")

        client.close()

    except FileNotFoundError as e:
        print(f"Error: One or more template files not found. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    store_template_in_db()
