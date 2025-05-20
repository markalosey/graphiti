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

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ..prompts.models import Message
from .client import MULTILINGUAL_EXTRACTION_RESPONSES, LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gpt-4.1-mini'
DEFAULT_SMALL_MODEL = 'gpt-4.1-nano'


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
    MAX_RETRIES: ClassVar[int] = 2

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: typing.Any = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the OpenAIClient with the provided configuration, cache setting, and client.

        Args:
            config (LLMConfig | None): The configuration for the LLM client, including API key, model, base URL, temperature, and max tokens.
            cache (bool): Whether to use caching for responses. Defaults to False.
            client (Any | None): An optional async client instance to use. If not provided, a new AsyncOpenAI client is created.

        """
        # removed caching to simplify the `generate_response` override
        if cache:
            raise NotImplementedError('Caching is not implemented for OpenAI')

        if config is None:
            config = LLMConfig()

        super().__init__(config, cache)

        if client is None:
            self.client = AsyncOpenAI(api_key=config.api_key, base_url=config.base_url)
        else:
            self.client = client

        self.max_tokens = max_tokens

    async def _generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        openai_messages: list[ChatCompletionMessageParam] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == 'user':
                openai_messages.append({'role': 'user', 'content': m.content})
            elif m.role == 'system':
                openai_messages.append({'role': 'system', 'content': m.content})

        try:
            if temperature is None:
                temperature = self.temperature
            if max_tokens is None:
                max_tokens = self.max_tokens

            if (
                response_model
                and hasattr(self.client, 'beta')
                and hasattr(self.client.beta.chat.completions, 'parse')
            ):
                # This is the path that was failing with llama_decode
                logger.critical('LLM_CLIENT: Using completions.parse() with response_model')
                try:
                    request_params = {
                        'model': self.model or DEFAULT_MODEL,
                        'messages': openai_messages,
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                    }
                    if response_model:
                        request_params['response_format'] = {'type': 'json_object'}

                    logger.critical(
                        f'LLM_REQUEST_PARAMS: {json.dumps(request_params, indent=2)}'
                    )  # Log the request

                    response = await self.client.beta.chat.completions.parse(
                        **request_params,
                        response_model=response_model,  # type: ignore[arg-type]
                    )
                    # If parse() returns a Pydantic model, convert to dict for consistency
                    # Or handle Pydantic model directly if downstream code expects it
                    # Assuming downstream expects a dict for now
                    if hasattr(response, 'model_dump'):
                        llm_response = response.model_dump()
                    else:
                        # This case should ideally not happen if parse() works as expected with response_model
                        logger.warning(
                            'completions.parse() did not return a Pydantic model. LLM output might be raw.'
                        )
                        llm_response = {'raw_response': str(response)}  # Fallback

                except Exception as e:
                    logger.error(
                        f'LLM_CLIENT: Error during completions.parse(): {str(e)}', exc_info=True
                    )
                    # Attempt to get raw response if parse fails, for debugging
                    try:
                        del request_params['response_format']  # Remove json_object for raw call
                        # Might need to remove other params not valid for a raw call
                        raw_completion = await self.client.chat.completions.create(**request_params)
                        raw_content = raw_completion.choices[0].message.content
                        logger.error(
                            f'LLM_CLIENT: Raw LLM response on parse() failure: {raw_content}'
                        )
                        llm_response = {
                            'error_during_parse': str(e),
                            'raw_llm_output_attempt': raw_content,
                        }
                    except Exception as raw_e:
                        logger.error(
                            f'LLM_CLIENT: Failed to get raw response after parse() error: {str(raw_e)}'
                        )
                        llm_response = {
                            'error_during_parse': str(e),
                            'raw_llm_output_attempt_failed': str(raw_e),
                        }
                    # Re-raise the original parsing/request error if you want the calling code to handle it
                    # For now, returning the error dict to see it in logs and potentially in tool response
            else:
                # Fallback or standard non-parsing path
                logger.critical('LLM_CLIENT: Using standard completions.create()')
                completion = await self.client.chat.completions.create(
                    model=self.model or DEFAULT_MODEL,
                    messages=openai_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw_content = completion.choices[0].message.content
                logger.debug(f'LLM Raw Content: {raw_content}')
                if response_model:
                    try:
                        llm_response = json.loads(raw_content or '{}')
                    except json.JSONDecodeError:
                        logger.error(f'Failed to parse LLM JSON output: {raw_content}')
                        llm_response = {
                            'error': 'LLM output was not valid JSON',
                            'raw_content': raw_content,
                        }
                else:
                    llm_response = {'content': raw_content}

            return llm_response
        except openai.LengthFinishReasonError as e:
            raise Exception(f'Output length exceeded max tokens {self.max_tokens}: {e}') from e
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f'Error in generating LLM response: {e}')
            raise

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        last_error = None

        # Add multilingual extraction instructions
        messages[0].content += MULTILINGUAL_EXTRACTION_RESPONSES

        while retry_count <= self.MAX_RETRIES:
            try:
                response = await self._generate_response(messages, response_model, max_tokens)
                return response
            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                # Let OpenAI's client handle these retries
                raise
            except Exception as e:
                last_error = e

                # Don't retry if we've hit the max retries
                if retry_count >= self.MAX_RETRIES:
                    logger.error(f'Max retries ({self.MAX_RETRIES}) exceeded. Last error: {e}')
                    raise

                retry_count += 1

                # Construct a detailed error message for the LLM
                error_context = (
                    f'The previous response attempt was invalid. '
                    f'Error type: {e.__class__.__name__}. '
                    f'Error details: {str(e)}. '
                    f'Please try again with a valid response, ensuring the output matches '
                    f'the expected format and constraints.'
                )

                error_message = Message(role='user', content=error_context)
                messages.append(error_message)
                logger.warning(
                    f'Retrying after application error (attempt {retry_count}/{self.MAX_RETRIES}): {e}'
                )

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')
