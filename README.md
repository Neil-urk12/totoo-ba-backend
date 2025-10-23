# AI RAG Product Checker

AI RAG Product Checker is a FastAPI-based service designed to verify FDA Philippines-regulated products using both database lookups and AI-powered image analysis. The application supports verification of food products, drug products, medical devices, cosmetics, and establishments.

## Features

- **ID-based Verification**: Check products by registration numbers, license numbers, or tracking numbers.
- **AI Image Verification**: Upload product images for AI-powered extraction and verification using Google's Gemini.
- **Fuzzy Matching**: Intelligent matching with multiple scoring algorithms.
- **Multi-category Support**: Handles drugs, food, medical devices, cosmetics, and establishments.
- **RESTful API**: Clean API endpoints with proper error handling and documentation.
- **Hybrid OCR**: A multi-layered OCR approach using Tesseract, Groq, and Gemini for fast and accurate text extraction.
- **High Performance**: Optimized with uvloop and httptools for 2-4x faster async I/O performance.

## Performance Optimizations

This application is optimized for high-performance async operations:

### uvloop + httptools Integration
- **uvloop**: Drop-in replacement for asyncio's event loop, providing 2-4x performance improvement
- **httptools**: Fast HTTP request/response parser written in Cython
- **Async Database Operations**: Using asyncpg for PostgreSQL with connection pooling
- **Fast JSON Serialization**: Using orjson for rapid JSON encoding/decoding

### Why This Matters for Your Use Case
- **AI Image Processing**: Faster I/O means quicker image uploads and processing
- **Database Queries**: Better handling of concurrent database lookups and fuzzy matching
- **External API Calls**: Improved performance when calling FDA verification endpoints
- **Concurrent Requests**: Better scalability for multiple simultaneous product verifications

### Benchmarks
Typical performance improvements with uvloop + httptools:
- **Request throughput**: 2-4x increase in requests per second
- **Response latency**: 30-50% reduction in average response time
- **Memory usage**: 10-20% lower memory footprint
- **CPU efficiency**: Better utilization of system resources

## Project Structure

```
├── app/
│   ├── api/
│   │   ├── deps.py
│   │   ├── products.py
│   │   └── repository/
│   │       ├── database_repository.py
│   │       └── products_repository.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   └── logging.py
│   ├── models/
│   ├── schemas/
│   ├── services/
│   │   ├── ocr_service.py
│   │   ├── product_verification_service.py
│   │   └── upsert_extractors/
│   └── utils/
├── bruno_api_testing/
├── .env.example
├── README.md
└── requirements.txt
```

- **app/api**: API endpoints and dependency injection.
- **app/core**: Core application settings, database configuration, and logging.
- **app/models**: SQLAlchemy database models.
- **app/schemas**: Pydantic schemas for data validation and serialization.
- **app/services**: Business logic, including product verification and OCR services.
- **app/utils**: Helper functions.
- **bruno_api_testing**: API tests using Bruno.

## Prerequisites

- Python 3.9 or higher
- PostgreSQL database
- Git
- pip (Python package installer)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file by copying the example:**
   ```bash
   cp .env.example .env
   ```

## Configuration

### Environment Variables

Update the `.env` file with your specific configuration:

```
# Database Configuration
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/product_checker

# Application Configuration
APP_NAME=AI RAG Product Checker
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=True

# Security Settings
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# CORS Configuration
CORS_ORIGINS=["http://localhost:5173", "http://localhost:8000"]

# FDA Scraper Configuration
FDA_BASE_URL=https://verification.fda.gov.ph
FDA_TIMEOUT=30
FDA_MAX_RETRIES=3
FDA_RATE_LIMIT_DELAY=1.0
FDA_MAX_PAGES_PER_RUN=10

# Business Databank Configuration
BUSINESS_DATABANK_URL=https://databank.business.gov.ph
SEC_API_URL=https://portal.sec.gov.ph
SEC_API_KEY=your-sec-api-key-here

# Fuzzy Matching Configuration
FUZZY_MATCH_THRESHOLD=80
FUZZY_MATCH_LIMIT=5

# Caching Configuration
CACHE_ENABLED=True
CACHE_TTL_MINUTES=30
CACHE_MAX_SIZE=1000

# Background Tasks Configuration
BACKGROUND_TASK_TIMEOUT=300

# Logging Configuration
LOG_LEVEL=INFO

# Groq AI Configuration
GROQ_API_KEY=your-groq-api-key-here
```

### Database Setup

1. Ensure PostgreSQL is installed and running
2. Create a database specified in your `DATABASE_URL`
3. The application will automatically create tables on startup

## Usage

### Running the Application

The application includes performance optimizations with **uvloop** and **httptools** for faster async I/O and HTTP parsing.

#### Quick Start (Recommended)

```bash
# Install dependencies
make install

# Run development server with optimizations
make dev

# Or run production server with optimizations  
make prod
```

#### Development Mode (with auto-reload)

```bash
# Using the optimized development runner
python run_dev.py

# Or traditional uvicorn with optimizations
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 --loop uvloop --http httptools
```

#### Production Mode

```bash
# Using the optimized production runner
python run_production.py

# Or traditional uvicorn with optimizations
uvicorn app.main:app --host 0.0.0.0 --port 8000 --loop uvloop --http httptools --workers 4
```

The application will be available at `http://localhost:8000`.

**API Documentation:**
- Interactive docs: `http://localhost:8000/docs`
- Alternative docs: `http://localhost:8000/redoc`

### API Endpoints

- `GET /` - Root endpoint with application info
- `GET /health` - Health check endpoint
- `GET /api/v1/products/verify/{product_id}` - Verify product by ID
- `POST /api/v1/products/verify-image` - Verify product from image upload

## Development

### Running Tests

To run tests (if available):

```bash
pytest
```

### Environment Variables for Development

For development, use the following settings in your `.env`:

```
ENVIRONMENT=development
DEBUG=True
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/product_checker_dev
CORS_ORIGINS=["http://localhost:5173", "http://localhost:8000"]
```

## Production Deployment

For production deployment, ensure the following:

- Set `ENVIRONMENT=production`
- Set `DEBUG=False`
- Use a strong `SECRET_KEY`
- Configure proper logging

Example production command:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

The API is documented using FastAPI's built-in documentation. After starting the server, visit:

- `http://localhost:8000/docs` for the interactive API documentation
- `http://localhost:8000/redoc` for the alternative API documentation

## Troubleshooting

### Common Issues

1. **Database Connection Issues**: Ensure PostgreSQL is running and your `DATABASE_URL` is correct.
2. **Dependency Issues**: Make sure you're using the virtual environment and have installed all requirements.
3. **AI Service Issues**: Verify your `GROQ_API_KEY` is set correctly and you have internet access to the Groq services.

### Error Codes

- `400`: Bad request - Invalid input data
- `404`: Not found - Product not found in database
- `422`: Unprocessable entity - Validation error
- `500`: Internal server error

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
6. Push to the branch (`git push origin feature/AmazingFeature`)
7. Open a Pull Request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- FastAPI for the excellent web framework
- SQLAlchemy for the ORM
- Groq's Llama models for AI capabilities
- All contributors to the open-source libraries used in this project
