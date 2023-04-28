# API Endpoints Documentation

This document provides detailed information on the request and response formats for each endpoint in the Plan Prediction Tool API.

## Table of Contents

- [`/get_similar_activities`](#get_similar_activities)
- [`/get_activity_suggestions`](#get_activity_suggestions)
- [`/get_initial_contribution_suggestions`](#get_initial_contribution_suggestions)
- [`/get_combined_contribution_suggestions`](#get_combined_contribution_suggestions)

## `/get_similar_activities`

Returns a list of similar activities. Has the number of nearest neighbors to return as an endpoint parameter *k*, defaults to 5 if not specified.

**Request body:**
```json
{
    "TITLE": string,
    "GROUP_RESPONSIBLE_NAME": string,
    "RESPONSIBLE_WITH_DETAILS": string,
    "ACTIVITY_TYPE_EN": string,
    "WBS_NODE_CODE": string,
    "CREATOR_NAME": string
}
```

These are the required fields (they can be null), but more activity information can be provided, with no influence on the results.

**Response:**
A list of the *k* most similar activities to the one provided. 

```json
{
        "ACTIVITY_TYPE_EN": string,
        "ACTIVITY_UUID": string,
        "CREATION_DATE": string,
        "CREATOR_NAME": string,
        "FACILITY_NAMES": string,
        "GOAL": string,
        "GROUP_RESPONSIBLE_NAME": string,
        "IMPACT_NOT_DONE": string,
        "LOCATION_INFORMATION": string,
        "PRIORITY_EN": string,
        "RESPONSIBLE_WITH_DETAILS": string,
        "TITLE": string,
        "WBS_NODE_CODE": string,
        "similarity_score": float
    }
```

The **similarity_score** field in the returned JSON object represents the similarity between the input activity and the returned similar activities on a scale of 0 to 1. A higher value indicates a higher similarity between the activities. Developers can use this score to determine a threshold for suggesting similar activities to users.

## `/get_activity_suggestions`

Returns suggestions for the missing fields in the current state of the new activity. Has the number of nearest neighbors to use for calculation *k* and the number of returned values for each missing field *max_list_size* as endpoint parameters, defaults to 10 and 5 respectively, if not specified.

**Request body:**
```json
{
    "TITLE": string,
    "GROUP_RESPONSIBLE_NAME": string,
    "RESPONSIBLE_WITH_DETAILS": string,
    "ACTIVITY_TYPE_EN": string,
    "WBS_NODE_CODE": string,
    "CREATOR_NAME": string
}
```

These are the required fields (they can be null), but more activity information can be provided, with no influence on the results.

**Response:**
A list of up to *max_list_size* ordered suggestions for each of the relevant fields that had missing values. For example, if RESPONSIBLE_WITH_DETAILS and WBS_NODE_CODE were null:

```json
{
        "RESPONSIBLE_WITH_DETAILS": [
            {
                "confidence": float,
                "value": string
            }
            ...
        ],
        "WBS_NODE_CODE": [
            {
                "confidence": float,
                "value": string
            }
            ...
        ]
    }
```

- **confidence**: the percentage of times this suggestion was present in *k* the nearest neighbors.
- **value:** the actual suggested value.

## `/get_initial_contribution_suggestions`

Returns suggested contributions based on the new activity. Has the number of nearest neighbors to use for calculation *k* and the number of suggested contributions for each phase *max_list_size* as endpoint parameters, defaults to 10 and 5 respectively, if not specified.

**Request body:**
```json
{
    "TITLE": string,
    "GROUP_RESPONSIBLE_NAME": string,
    "RESPONSIBLE_WITH_DETAILS": string,
    "ACTIVITY_TYPE_EN": string,
    "WBS_NODE_CODE": string,
    "CREATOR_NAME": string
}
```

These are the required fields (they can be null), but more activity information can be provided, with no influence on the results.

**Response:**
A list of up to *max_list_size* ordered suggestions for contribution requests to make for each phase:

```json
{
"Commissioning": [
        {
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string,
            "confidence": float
        },
        ...
    ],
    "Installation": [
        {
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string,
            "confidence": float
        },
        ...
    ],
    "Preparation": [
        {
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string,
            "confidence": float
        },
        ...
    ]
    }
```

- **confidence**: the percentage of times this contribution suggestion was present in the *k* nearest neighbors' contributions.

## `/get_combined_contribution_suggestions`

Returns suggested contributions based on the new activity, but also on the contributions the user has already entered. Has the number of nearest neighbors to use for calculation *k* and the number of suggested contributions for each phase *max_list_size* as endpoint parameters, defaults to 10 and 5 respectively, if not specified.

**Request body:**
```json
    {
    "activity": {
        "TITLE": string,
        "GROUP_RESPONSIBLE_NAME": string,
        "RESPONSIBLE_WITH_DETAILS": string,
        "ACTIVITY_TYPE_EN": string,
        "WBS_NODE_CODE": string,
        "CREATOR_NAME": string
    },
    "confirmed_contributions": [
        {
            "PHASE_NAME": string,
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string
        },
        ...
    ]
}
```

**Response:**
A list of up to *max_list_size* ordered suggestions for contribution requests to make for each phase. The suggestions from the association rules come first, even if they have lesser confidence, because they come from actual entered contributions by the user:

```json
{
"Commissioning": [
        {
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string,
            "confidence": float
        },
        ...
    ],
    "Installation": [
        {
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string,
            "confidence": float
        },
        ...
    ],
    "Preparation": [
        {
            "CONTRIBUTION_TYPE": string,
            "ORG_UNIT_CODE": string,
            "confidence": float
        },
        ...
    ]
    }
```

- **confidence**: the percentage of times this contribution suggestion was present in the *k* nearest neighbors' contributions, or the confidence of the association rule

