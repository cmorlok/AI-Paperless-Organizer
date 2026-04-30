"""Pydantic request/response models for the classifier router."""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ClassifierConfigUpdate(BaseModel):
    enable_title: Optional[bool] = None
    enable_tags: Optional[bool] = None
    enable_correspondent: Optional[bool] = None
    enable_document_type: Optional[bool] = None
    enable_storage_path: Optional[bool] = None
    enable_created_date: Optional[bool] = None
    enable_custom_fields: Optional[bool] = None
    tag_behavior: Optional[str] = None
    correspondent_behavior: Optional[str] = None
    review_mode: Optional[str] = None
    batch_size: Optional[int] = None
    prompt_title: Optional[str] = None
    prompt_tags: Optional[str] = None
    prompt_correspondent: Optional[str] = None
    prompt_document_type: Optional[str] = None
    prompt_date: Optional[str] = None
    system_prompt: Optional[str] = None
    tags_min: Optional[int] = None
    tags_max: Optional[int] = None
    tags_keep_existing: Optional[bool] = None
    tags_ignore: Optional[List[str]] = None
    tags_protected: Optional[List[str]] = None
    dates_ignore: Optional[List[str]] = None
    storage_path_behavior: Optional[str] = None
    storage_path_override_names: Optional[List[str]] = None
    excluded_tag_ids: Optional[List[int]] = None
    excluded_correspondent_ids: Optional[List[int]] = None
    excluded_document_type_ids: Optional[List[int]] = None
    correspondent_trim_prompt: Optional[bool] = None
    correspondent_strip_legal: Optional[bool] = None
    correspondent_ignore: Optional[List[str]] = None
    auto_classify_enabled: Optional[bool] = None
    auto_classify_interval: Optional[int] = None
    auto_classify_mode: Optional[str] = None
    auto_classify_skip_tag_ids: Optional[List[int]] = None
    classification_tag_enabled: Optional[bool] = None
    classification_tag_name: Optional[str] = None
    review_tag_enabled: Optional[bool] = None
    review_tag_name: Optional[str] = None
    tag_ideas_tag_enabled: Optional[bool] = None
    tag_ideas_tag_name: Optional[str] = None


class StoragePathProfileUpdate(BaseModel):
    paperless_path_id: int
    paperless_path_name: str = ""
    paperless_path_path: str = ""
    enabled: bool = True
    person_name: str = ""
    path_type: str = "private"
    context_prompt: str = ""


class CustomFieldMappingUpdate(BaseModel):
    paperless_field_id: int
    paperless_field_name: str = ""
    paperless_field_type: str = "string"
    enabled: bool = False
    extraction_prompt: str = ""
    example_values: str = ""
    validation_regex: str = ""
    ignore_values: str = ""


class ApplyRequest(BaseModel):
    document_id: int
    classification: Dict[str, Any]


class BenchmarkSlot(BaseModel):
    provider: str = "openai"
    model: str = ""


class BenchmarkRequest(BaseModel):
    document_id: int
    slots: List[BenchmarkSlot]


class TagIdeaApproveRequest(BaseModel):
    tag_name: str
