import asyncio
from contextlib import asynccontextmanager
from functools import partial
import logging

from fastapi import APIRouter, FastAPI, status
from graphiti_core.nodes import EpisodeType  # type: ignore
from graphiti_core.utils.maintenance.graph_data_operations import clear_data  # type: ignore

from graph_service.dto import AddEntityNodeRequest, AddMessagesRequest, Message, Result
from graph_service.zep_graphiti import ZepGraphitiDep, ENTITY_TYPES

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AsyncWorker:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.task = None

    async def worker(self):
        while True:
            try:
                job = await self.queue.get()
                await job()
            except asyncio.CancelledError:
                logger.info('AsyncWorker task cancelled.')
                break
            except Exception as e:
                logger.error(f'AsyncWorker error in job: {str(e)}', exc_info=True)
            finally:
                pass

    async def start(self):
        self.task = asyncio.create_task(self.worker())
        logger.info('AsyncWorker started.')

    async def stop(self):
        logger.info('AsyncWorker stopping...')
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                logger.info('AsyncWorker task successfully cancelled during stop.')
        while not self.queue.empty():
            self.queue.get_nowait()
        logger.info('AsyncWorker queue cleared and stopped.')


async_worker = AsyncWorker()


@asynccontextmanager
async def lifespan(_: FastAPI):
    await async_worker.start()
    yield
    await async_worker.stop()


router = APIRouter(lifespan=lifespan)


@router.post('/messages', status_code=status.HTTP_202_ACCEPTED)
async def add_messages(
    request: AddMessagesRequest,
    graphiti: ZepGraphitiDep,
):
    async def add_messages_task(m: Message):
        logger.critical(
            f'!!!!!!!!!!!! ASYNC_WORKER: add_messages_task started for message: {m.name} - {m.uuid} !!!!!!!!!!!!'
        )
        try:
            await graphiti.add_episode(
                uuid=m.uuid,
                group_id=request.group_id,
                name=m.name,
                episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
                reference_time=m.timestamp,
                source=EpisodeType.message,
                source_description=m.source_description,
                entity_types=ENTITY_TYPES,
            )
            logger.critical(
                f'!!!!!!!!!!!! ASYNC_WORKER: add_messages_task COMPLETED for message: {m.name} - {m.uuid} !!!!!!!!!!!!'
            )
        except Exception as e:
            logger.critical(
                f'!!!!!!!!!!!! ASYNC_WORKER: EXCEPTION in add_messages_task for {m.name} - {m.uuid}: {str(e)} !!!!!!!!!!!!',
                exc_info=True,
            )

    for m_idx, m_message in enumerate(request.messages):
        logger.info(
            f"Queuing message {m_idx + 1} of {len(request.messages)}: Name='{m_message.name}', UUID='{m_message.uuid}'"
        )
        await async_worker.queue.put(partial(add_messages_task, m_message))

    return Result(message='Messages added to processing queue', success=True)


@router.post('/entity-node', status_code=status.HTTP_201_CREATED)
async def add_entity_node(
    request: AddEntityNodeRequest,
    graphiti: ZepGraphitiDep,
):
    node = await graphiti.save_entity_node(
        uuid=request.uuid,
        group_id=request.group_id,
        name=request.name,
        summary=request.summary,
    )
    return node


@router.delete('/entity-edge/{uuid}', status_code=status.HTTP_200_OK)
async def delete_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_entity_edge(uuid)
    return Result(message='Entity Edge deleted', success=True)


@router.delete('/group/{group_id}', status_code=status.HTTP_200_OK)
async def delete_group(group_id: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_group(group_id)
    return Result(message='Group deleted', success=True)


@router.delete('/episode/{uuid}', status_code=status.HTTP_200_OK)
async def delete_episode(uuid: str, graphiti: ZepGraphitiDep):
    await graphiti.delete_episodic_node(uuid)
    return Result(message='Episode deleted', success=True)


@router.post('/clear', status_code=status.HTTP_200_OK)
async def clear(
    graphiti: ZepGraphitiDep,
):
    await clear_data(graphiti.driver)
    await graphiti.build_indices_and_constraints()
    return Result(message='Graph cleared', success=True)
