from src.learning_db import fetch_unembedded, mark_embedded
from src.retrieval import Retriever
from src.logger import get_logger
logger = get_logger('retrain')

def main(batch=100):
    rows = fetch_unembedded(batch)
    if not rows:
        logger.info('No new feedback to embed.')
        return
    ret = Retriever()
    done_ids = []
    for fid, query, code in rows:
        try:
            ret.ensure_example(query, code, status='good')
            done_ids.append(fid)
        except Exception as e:
            logger.warning(f'Could not embed feedback {fid}: {e}')
    mark_embedded(done_ids)
    logger.info(f'Embedded and marked {len(done_ids)} examples.')

if __name__ == '__main__':
    main()