"""Model adapter registry for unified benchmark predictions."""

from .base import AdapterResult, BaseAdapter, build_prediction_record
from .registry import get_adapter

__all__ = ["AdapterResult", "BaseAdapter", "build_prediction_record", "get_adapter"]

