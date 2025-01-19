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
   - The resulting coordinates, along with the image hash and label, are cached for future use.

4. **Cache Structure**:
   - Key: Combination of image hash and label.
   - Value: Coordinates (array of coordinates).

5. **Response**:
   - If the label matches the cache, the cached coordinates are returned.
   - If no match is found, the pipeline generates new coordinates and returns them while updating the cache.
   - In case the instruction is irrelevant or the cv function is unable to find that particular keyword, it returns an empty array which is not hashed.

---

That’s it! You’re now ready to use the Visual QA pipeline.
