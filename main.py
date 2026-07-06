import uvicorn
import config
from api import app

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
    )
    uvicorn.run("api:app", host="0.0.0.0", port=config.settings.port, reload=False)
