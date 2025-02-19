import time
import threading
import math
import json
from metagpt.utils.token_counter import count_input_tokens
from metagpt.configs.llm_config import LLMConfig
from metagpt.logs import logger
from metagpt.configs.models_config import ModelsConfig
class RateLimitor:
    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self.tpm_bucket = TokenBucket(tpm)
        self.rpm_bucket = TokenBucket(rpm)
        self.lock = threading.RLock()
    
    def acquire_rpm(self, tokens=1):
        self.rpm_bucket.acquire(tokens)

    def cost_token(self, usage: dict):
        if not isinstance(usage, dict):
            usage = dict(usage)
        self.tpm_bucket._cost(usage.get("input_tokens", usage.get('prompt_tokens', 0)))
        self.tpm_bucket._cost(usage.get("output_tokens", usage.get('completion_tokens', 0)))
        if self.rpm > 0 or self.tpm > 0:
            self.lock.release()

    def acquire(self, messages):
        if self.rpm > 0 or self.tpm > 0:
            self.lock.acquire()
        tokens = count_input_tokens(messages)
        if self.tpm_bucket._wait(tokens):
            self.acquire_rpm(1)


class TokenBucket:
    def __init__(self, rpm):
        """
        Initialize the token bucket (thread-safe version)
        :param rpm: the number of requests per minute
        """
        self.capacity = rpm        # the capacity of the bucket
        self.tokens = rpm          # the current number of tokens
        self.rate = rpm / 60.0 if rpm else 0  # the number of tokens generated per second
        self.last_refill = time.time()
        self.lock = threading.RLock()  # 线程安全锁
        self.cond = threading.Condition(self.lock)  # 条件变量

    def _refill(self):
        """Refill the tokens (need to be called in the lock protected context)"""
        if self.capacity is None or self.capacity <= 0:
            return
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.rate
        
        if new_tokens > 0:
            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill = now
            return True  # 表示有新增令牌
        return False
    
    def _cost(self, tokens: int):
        if self.capacity is None or self.capacity <= 0:
            return
        assert tokens >= 0
        self._refill()
        self.tokens -= tokens

    def _wait(self, tokens: int):
        with self.cond:
            while True:
                if self.available_tokens >= tokens:
                    # enough tokens, return immediately
                    return True
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate

                logger.warning(f"current [{threading.current_thread().ident}] with [{self.tokens:.5f}] tokens, wait_time for tpm: {wait_time:.3f}")
                self.cond.wait(wait_time)

    def acquire(self, tokens=1):
        """
        Block until acquiring the specified number of tokens
        :param tokens: the number of tokens needed (default is 1)
        """
        if self.capacity is None or self.capacity <= 0:
            return
        with self.cond:
            while True:
                # try to refill the tokens in each loop
                self._refill()
                
                # if the tokens are enough, return immediately
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                
                # calculate the time to wait
                deficit = tokens - self.tokens
                wait_time = deficit / self.rate

                logger.warning(f"current [{threading.current_thread().ident}] with [{self.tokens:.5f}] tokens, wait_time for rpm: {wait_time:.3f}")
                
                # wait until the tokens are replenished (with timeout and notification)
                self.cond.wait(wait_time)

    @property
    def available_tokens(self):
        """Get the current number of available tokens (refreshed in real time)"""
        if self.capacity is None or self.capacity <= 0:
            return math.inf
        with self.lock:
            self._refill()
            return self.tokens
        

class RateLimitorRegistry:
    def __init__(self):
        self.rate_limitors = {}
        self.config_items = {}

    def init_rate_limitors(self):
        for model_name, llm_config in ModelsConfig.default().items():
            self.register(model_name, llm_config)

    def _config_to_key(self, llm_config: LLMConfig):
        return json.dumps(llm_config.model_dump())

    def register(self, model_name: str, llm_config: LLMConfig):
        if not llm_config:
            raise ValueError("llm_config is required")
        if not model_name:
            model_name = self._config_to_key(llm_config)
        if model_name not in self.rate_limitors:
            self.rate_limitors[model_name] = RateLimitor(llm_config.rpm, llm_config.tpm)
            self.config_items[self._config_to_key(llm_config)] = model_name
        return self.rate_limitors[model_name]
    
    def get(self, model_name: str):
        if not model_name:
            model_name = "_default_llm"
        return self.rate_limitors.get(model_name)
    
    def get_by_config(self, llm_config: LLMConfig):
        return self.rate_limitors[self._config_to_key(llm_config)]

rate_limitor_registry = RateLimitorRegistry()