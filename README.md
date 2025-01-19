# Visual QA Pipeline

## Project Description
This project implements a Visual QA pipeline that uses a combination of cryptographic hashing, LLMs, and computer vision to process user instructions and retrieve or compute the corresponding screen element coordinates. The pipeline efficiently caches results for future interactions.

---

## Table of Contents
- [Project Setup](#project-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Optimisations and Improvements](#optimisations-and-improvements)

---

## Project Setup

### Prerequisites
1. **Python 3.12.3**
   - Verify your Python version:
     ```bash
     python --version
     ```
     If the version is not 3.12.3, use `pyenv` to install the appropriate version.

2. **Redis**
   - Ensure a Redis server is running locally.
   - On macOS, you can start Redis using:
     ```bash
     redis-cli
     ```

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/atharva123a/visual-qa
   ```

2. Navigate into the project directory:
   ```bash
   cd visual-qa
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Environment Variables
Set up a `.env` file in the root directory with the following key:
```env
OPENAI_API_KEY=your-key
```

---

## Running the Application
1. Ensure your Redis server is running.
2. Start the application using `uvicorn`:
   ```bash
   uvicorn app.main:app --reload
   ```

---

## Usage
To test the implementation, run the following example:

### Example Request
```bash
curl --location 'http://127.0.0.1:8000/process-request' \
--form 'file=@"/Users/atharva/Downloads/sample_zepto.jpg"' \
--form 'instruction="Toggle the cafe section"'
```

### Sample Response
```json
{
  "message": "Request processed successfully",
  "coordinates": [277, 666, 69, 23]
}
```

---

## How It Works

1. **Image Hashing**:
   - The binary data for each image is hashed using a cryptographic hash function to generate a unique identifier.

2. **Instruction Processing**:
   - User instructions are passed to an LLM to extract a relevant keyword or label.
   - The extracted label is matched against cached data for the specific image hash.

3. **Cache Fallback**:
   - If no match is found, a computer vision function is used to detect the relevant screen element coordinates.
   - In case there are no labels that match with the extracted text from the image, we pass it to another function that uses YOLO to match target text with shapes that it extracted from the image.
   - The resulting coordinates, along with the image hash and label, are cached for future use.
   - There is also another bool stored to track whether the hash was generated based on textract extraction or yolo extraction.

4. **Cache Structure**:
   - Key: Combination of image hash and label.
   - Value: Json of Coordinates (array of coordinates) and whether it was detected using the vision model or not.

5. **Response**:
   - If the label matches the cache, the cached coordinates are returned.
   - If no match is found, the pipeline generates new coordinates and returns them while updating the cache.

---

## Optimisations and Improvements

### Current Implementation
- Right now, all labels that are matched on a particular screen are passed to the LLM to figure out a similarity match.
- The CV function only returns coordinates that exactly match with the extracted text from Tesseract.

### Future Optimisations
1. **Pre-filtering Labels Before LLM Matching**:
   - To improve performance, run a similarity check **before** invoking the LLM.
   - Retrieve the top 10 matching labels using:
     - An index-based similarity search mechanism (e.g., FAISS).
     - A smaller, lightweight model for quick similarity scoring in smaller use cases.
   - Pass only the top 10 most relevant labels to the LLM to fetch the most accurate match.
   
   This optimisation reduces the load on the LLM and enhances response times, especially in cases with a large number of cached labels.

2. **Improving Text-Instruction Matching for Coordinates**:
   - Run a similarity check between the extracted text and the instruction keyword text.
   - Based on the similarity scores, return the coordinates if the score exceeds a certain threshold (e.g., > 0.8).

   This ensures that text extraction errors or slight mismatches do not prevent relevant coordinates from being returned.
