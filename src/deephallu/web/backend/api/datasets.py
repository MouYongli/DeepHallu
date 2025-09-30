from fastapi import APIRouter, HTTPException, Query, Path
from fastapi.responses import FileResponse
from typing import Optional
import os

from deephallu.web.backend.services.dataset_service import DatasetService
from deephallu.web.backend.models.dataset_models import (
    DatasetListResponse, DatasetSamplesResponse, DatasetStatsResponse,
    DatasetBrowseRequest
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
dataset_service = DatasetService()


@router.get("/", response_model=DatasetListResponse)
async def get_datasets():
    """Get list of all available datasets"""
    try:
        datasets = dataset_service.get_all_datasets()
        return DatasetListResponse(
            datasets=datasets,
            total_count=len(datasets)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{dataset_name}/image/{image_id}", response_model=DatasetSamplesResponse)
async def browse_dataset_samples(
    dataset_name: str = Path(..., description="Name of the dataset"),
    image_id: str = Path(..., description="ID of the image"),
    search_query: Optional[str] = Query(None, description="Search in questions or image names")
):
    """Browse dataset samples with pagination and filtering"""
    try:
        request = DatasetBrowseRequest(
            dataset_name=dataset_name,
            page=page,
            page_size=page_size,
            category=category,
            search_query=search_query
        )
        return dataset_service.browse_dataset(request)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(
    dataset_name: str = Path(..., description="Name of the dataset")
):
    """Get statistics for a dataset"""
    try:
        return dataset_service.get_dataset_stats(dataset_name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/samples/{sample_id}/image")
async def get_sample_image(
    dataset_name: str = Path(..., description="Name of the dataset"),
    sample_id: str = Path(..., description="ID of the sample")
):
    """Get image file for a specific sample"""
    try:
        image_path = dataset_service.get_image(dataset_name, sample_id)
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")

        return FileResponse(
            image_path,
            media_type="image/jpeg",
            headers={"Cache-Control": "max-age=3600"}
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{dataset_name}/categories")
async def get_dataset_categories(
    dataset_name: str = Path(..., description="Name of the dataset")
):
    """Get available categories for a dataset"""
    try:
        dataset_config = dataset_service.config.get_dataset_config(dataset_name)
        if not dataset_config:
            raise HTTPException(status_code=404, detail="Dataset not found")

        return {
            "dataset_name": dataset_name,
            "categories": dataset_config.categories or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))