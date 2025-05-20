"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import typing
from typing import ClassVar, List, Dict, Optional, Type, Any
import json
import asyncio

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message as PromptMessage
from .client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'
DEFAULT_SMALL_MODEL = 'gpt-4.1-nano'
DEFAULT_MAX_TOKENS_INIT = 8192
DEFAULT_TEMPERATURE_INIT = 0.7
DEFAULT_MAX_RETRIES_INIT = 2
DEFAULT_RETRY_DELAY_INIT = 1.0


class OpenAIClient(LLMClient):
    """
    OpenAIClient is a client class for interacting with OpenAI's language models.

    This class extends the LLMClient and provides methods to initialize the client,
    get an embedder, and generate responses from the language model.

    Attributes:
        client (AsyncOpenAI): The OpenAI client used to interact with the API.
        model (str): The model name to use for generating responses.
        temperature (float): The temperature to use for generating responses.
        max_tokens (int): The maximum number of tokens to generate in a response.

    Methods:
        __init__(config: LLMConfig | None = None, cache: bool = False, client: typing.Any = None):
            Initializes the OpenAIClient with the provided configuration, cache setting, and client.

        _generate_response(messages: list[Message]) -> dict[str, typing.Any]:
            Generates a response from the language model based on the provided messages.
    """

    # Class-level constants
    MAX_RETRIES: ClassVar[int] = DEFAULT_MAX_RETRIES_INIT

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        cache: bool = False,
        client: Optional[AsyncOpenAI] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
        small_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.
            max_tokens (int): The maximum number of tokens to generate in a response.
            model (str): The model name to use for generating responses.
            temperature (float): The temperature to use for generating responses.
            max_retries (int): The maximum number of retries for generating responses.
            retry_delay (float): The delay between retries in seconds.
            small_model (str): The small model name to use for generating responses.

        """
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        effective_config = config if config is not None else LLMConfig()

        super().__init__(effective_config, cache)

        final_api_key = api_key or effective_config.api_key
        final_base_url = base_url or effective_config.base_url

        if client is None:
            self.client = AsyncOpenAI(api_key=final_api_key, base_url=final_base_url)
        else:
            self.client = client
            if final_api_key and (not self.client.api_key or self.client.api_key != final_api_key):
                self.client.api_key = final_api_key
            if final_base_url and (
                not self.client.base_url or str(self.client.base_url) != final_base_url
            ):
                self.client.base_url = final_base_url

        self.model = model or effective_config.model or DEFAULT_MODEL
        self.temperature = (
            temperature
            if temperature is not None
            else (
                effective_config.temperature
                if effective_config.temperature is not None
                else DEFAULT_TEMPERATURE_INIT
            )
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else (
                effective_config.max_tokens
                if effective_config.max_tokens is not None
                else DEFAULT_MAX_TOKENS_INIT
            )
        )
        self.max_retries = (
            max_retries
            if max_retries is not None
            else (
                effective_config.max_retries
                if hasattr(effective_config, 'max_retries')
                and effective_config.max_retries is not None
                else self.MAX_RETRIES
            )
        )
        self.retry_delay = (
            retry_delay
            if retry_delay is not None
            else (
                effective_config.retry_delay
                if hasattr(effective_config, 'retry_delay')
                and effective_config.retry_delay is not None
                else DEFAULT_RETRY_DELAY_INIT
            )
        )
        self.small_model = small_model or (
            effective_config.small_model
            if hasattr(effective_config, 'small_model') and effective_config.small_model
            else DEFAULT_SMALL_MODEL
        )

    async def _generate_response(
        self,
        messages: List[PromptMessage],
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens_override: Optional[int] = None,
        temperature_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'assistant':
                openai_messages.append({'role': 'assistant', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        effective_model = self.model
        effective_temperature = (
            temperature_override if temperature_override is not None else self.temperature
        )
        effective_max_tokens = (
            max_tokens_override if max_tokens_override is not None else self.max_tokens
        )

        request_params = {
            'model': effective_model,
            'messages': openai_messages,
            'temperature': effective_temperature,
            'max_tokens': effective_max_tokens,
        }

        raw_content = None
        llm_response = {}

        logger.critical(f'LLM_REQUEST_PARAMS: {json.dumps(request_params, indent=2)}')

        try:
            if response_model:
                logger.critical(
                    'LLM_CLIENT: Attempting structured output via create() and model_validate_json() (REMOVING response_format hint if present)'
                )
                if 'response_format' in request_params:
                    del request_params['response_format']
                logger.critical(
                    f'LLM_REQUEST_PARAMS (for structured): {json.dumps(request_params, indent=2)}'
                )

                completion = await self.client.chat.completions.create(**request_params)
                raw_content = completion.choices[0].message.content
                logger.critical(f'LLM_CLIENT: Raw content for structured parsing: {raw_content}')

                if raw_content:
                    parsed_model = response_model.model_validate_json(raw_content)
                    llm_response = parsed_model.model_dump()
                else:
                    logger.error('LLM_CLIENT: Received empty content for structured parsing.')
                    llm_response = {'error': 'LLM returned empty content for structured parsing'}
            else:
                logger.critical(
                    'LLM_CLIENT: Using standard completions.create() for non-structured output'
                )
                if 'response_format' in request_params:
                    del request_params['response_format']
                logger.critical(
                    f'LLM_REQUEST_PARAMS (for non-structured): {json.dumps(request_params, indent=2)}'
                )
                completion = await self.client.chat.completions.create(**request_params)
                raw_content = completion.choices[0].message.content
                logger.debug(f'LLM Raw Content: {raw_content}')
                llm_response = {'content': raw_content}

        except openai.BadRequestError as e:
            logger.error(f'LLM_CLIENT: OpenAI BadRequestError: {str(e)}', exc_info=True)
            error_body = 'Unknown (response not available)'
            if e.response and hasattr(e.response, 'text') and e.response.text:
                error_body = e.response.text
            elif e.response and hasattr(e.response, 'json') and callable(e.response.json):
                try:
                    error_body = e.response.json()
                except:
                    pass
            logger.error(f'LLM_CLIENT: OpenAI BadRequestError body: {error_body}')
            llm_response = {
                'error': f'OpenAI API BadRequestError: {str(e)}',
                'error_body': error_body,
            }
            if 'llama_decode' in str(e).lower() or (
                isinstance(error_body, str) and 'llama_decode' in error_body.lower()
            ):
                logger.error('LLM_CLIENT: Encountered llama_decode error.')
        except json.JSONDecodeError as e_json:
            logger.error(
                f'LLM_CLIENT: JSONDecodeError while parsing LLM response: {str(e_json)}. Raw content: {raw_content}',
                exc_info=True,
            )
            llm_response = {'error': 'Failed to parse LLM JSON output', 'raw_content': raw_content}
        except Exception as e_generic:
            logger.error(
                f'LLM_CLIENT: Unexpected error during LLM call or parsing: {str(e_generic)}',
                exc_info=True,
            )
            raw_fallback_content = (
                raw_content if raw_content is not None else 'Not available from initial attempt'
            )
            llm_response = {
                'error': f'Unexpected error in LLM processing: {str(e_generic)}',
                'raw_content_on_error': raw_fallback_content,
            }

        return llm_response

    async def generate_response(
        self,
        messages: List[PromptMessage],
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        model_size: Optional[ModelSize] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_exception = None

        while retry_count <= self.max_retries:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, temperature
                )
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except openai.RateLimitError as e:
                last_exception = e
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f'Max retries ({self.max_retries}) exceeded for RateLimitError.')
                    break
                sleep_time = self.retry_delay * (2**retry_count)
                logger.warning(
                    f'RateLimitError. Retrying in {sleep_time} seconds... (attempt {retry_count}/{self.max_retries})'
                )
                await asyncio.sleep(sleep_time)
            except openai.APIError as e:
                last_exception = e
                retry_count += 1
                if retry_count > self.max_retries:
                    logger.error(f'Max retries ({self.max_retries}) exceeded for APIError.')
                    break
                sleep_time = self.retry_delay * (2**retry_count)
                logger.warning(
                    f'APIError: {str(e)}. Retrying in {sleep_time} seconds... (attempt {retry_count}/{self.max_retries})'
                )
                await asyncio.sleep(sleep_time)
            except Exception as e:
                logger.error(
                    f'Unhandled exception from _generate_response: {str(e)}', exc_info=True
                )
                raise e

        if last_exception:
            raise last_exception
        else:
            raise Exception('LLM call failed after retries without specific exception.')
