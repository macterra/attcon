"""API-only explicit literal types for the frozen relational report schema.

This module is inert on import and uses only stdlib. It does not alter the v1
exporter, scientific inputs, report contract, auditors, or any preserved run.
Provider acceptance must be established separately; this is not an API client.
"""
from copy import deepcopy
import hashlib
import json


SOURCE_SCHEMA_CANONICAL_SHA256 = (
    'c9d03e517b61f35664506d416f62fb852355bb242ef890485feb9f84c25d9a41'
)
ADDED_TYPE_COUNT = 22
STRING_TYPE_COUNT = 19
NULLABLE_STRING_TYPE_COUNT = 3


def build_typed_response_schema(source_schema):
    """Deep-copy the exact frozen schema and add only 22 implied leaf types.

    The schema digest pins the complete input rather than allowing this narrow
    compatibility amendment to silently transform a different report contract.
    Existing enum/const values, required arrays, bounds and ordering are copied
    verbatim. Nullable command enums retain their existing null member.
    """
    canonical = json.dumps(source_schema, sort_keys=True, separators=(',', ':'),
                           allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != SOURCE_SCHEMA_CANONICAL_SHA256:
        raise ValueError('typed-schema amendment requires the exact frozen v1 schema')
    amended = deepcopy(source_schema)
    counts = {'string': 0, 'nullable_string': 0}

    def visit(node):
        if type(node) is dict:
            if 'type' not in node and ('const' in node or 'enum' in node):
                values = [node['const']] if 'const' in node else node['enum']
                if not values or any(value is not None and type(value) is not str
                                     for value in values):
                    raise ValueError('unexpected literal domain in frozen schema')
                if None in values:
                    node['type'] = ['string', 'null']
                    counts['nullable_string'] += 1
                else:
                    node['type'] = 'string'
                    counts['string'] += 1
            for child in node.values():
                visit(child)
        elif type(node) is list:
            for child in node:
                visit(child)

    visit(amended)
    if counts != {'string': STRING_TYPE_COUNT,
                  'nullable_string': NULLABLE_STRING_TYPE_COUNT}:
        raise ValueError('unexpected number of explicit literal type additions')
    return amended
