import logging
from typing import Annotated, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, status

from graph_service.zep_graphiti import Graphiti, get_graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix='/graph',
    tags=['Graph Administration'],
)


@router.post(
    '/clear',
    status_code=status.HTTP_200_OK,
    summary='Clear all data from the graph',
    description="""Completely clears all nodes and relationships from the Neo4j database.
    This operation is irreversible. It also attempts to clear and rebuild search indices.""",
)
async def clear_graph_data(
    graphiti_client: Annotated[Graphiti, Depends(get_graphiti)],
) -> Dict[str, Any]:
    try:
        logger.info('Attempting to clear all graph data...')
        # Clear graph data using the imported function and the client's driver
        await clear_data(graphiti_client.driver)
        logger.info('Graph data cleared successfully.')

        # Rebuild indices using the method on the graphiti_client itself
        logger.info('Attempting to clear and rebuild search indices...')
        # Assuming we want to delete existing for a full clear; the method takes delete_existing
        await graphiti_client.build_indices_and_constraints(delete_existing=True)
        logger.info('Search indices cleared and rebuilt successfully.')

        return {'message': 'Graph data and search indices cleared and rebuilt successfully.'}
    except Exception as e:
        logger.error(f'Error clearing graph data or rebuilding indices: {e}', exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'An error occurred: {str(e)}',
        )
