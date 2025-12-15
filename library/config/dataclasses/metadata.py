from dataclasses import dataclass, field
from typing import Optional

@dataclass
class MetadataConfig:
    metadata_title: Optional[str] = field(default=None, metadata={"help": "title for model metadata (default is output_name)"})
    metadata_author: Optional[str] = field(default=None, metadata={"help": "author name for model metadata"})
    metadata_description: Optional[str] = field(default=None, metadata={"help": "description for model metadata"})
    metadata_license: Optional[str] = field(default=None, metadata={"help": "license for model metadata"})
    metadata_tags: Optional[str] = field(default=None, metadata={"help": "tags for model metadata, separated by comma"})
    metadata_usage_hint: Optional[str] = field(default=None, metadata={"help": "usage hint for model metadata"})
    metadata_thumbnail: Optional[str] = field(default=None, metadata={"help": "thumbnail image as data URL or file path (will be converted to data URL) for model metadata"})
    metadata_merged_from: Optional[str] = field(default=None, metadata={"help": "source models for merged model metadata"})
    metadata_trigger_phrase: Optional[str] = field(default=None, metadata={"help": "trigger phrase for model metadata"})
    metadata_preprocessor: Optional[str] = field(default=None, metadata={"help": "preprocessor used for model metadata"})
    metadata_is_negative_embedding: Optional[str] = field(default=None, metadata={"help": "whether this is a negative embedding for model metadata"})
