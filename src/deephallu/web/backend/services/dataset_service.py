import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image

from ..core.config import config
from ..models.dataset_models import (
    DatasetInfo, DatasetSample, DatasetSamplesResponse,
    DatasetStatsResponse, DatasetBrowseRequest
)


class DatasetService:
    def __init__(self):
        self.config = config

    def get_all_datasets(self) -> List[DatasetInfo]:
        datasets = []
        for name, dataset_config in self.config.get_all_datasets().items():
            # Count total samples
            total_samples = self._count_dataset_samples(name, dataset_config.path)

            dataset_info = DatasetInfo(
                name=dataset_config.name,
                type=dataset_config.type,
                description=dataset_config.description,
                path=dataset_config.path,
                categories=dataset_config.categories,
                total_samples=total_samples
            )
            datasets.append(dataset_info)
        return datasets

    def browse_dataset(self, request: DatasetBrowseRequest) -> DatasetSamplesResponse:
        dataset_config = self.config.get_dataset_config(request.dataset_name)
        if not dataset_config:
            raise ValueError(f"Dataset '{request.dataset_name}' not found")

        # Load dataset samples based on type
        all_samples = self._load_dataset_samples(request.dataset_name, dataset_config)

        # Filter by category if specified
        if request.category and dataset_config.categories:
            all_samples = [s for s in all_samples if s.category == request.category]

        # Filter by search query if specified
        if request.search_query:
            query_lower = request.search_query.lower()
            all_samples = [
                s for s in all_samples
                if (s.question and query_lower in s.question.lower()) or
                   (s.image_name and query_lower in s.image_name.lower())
            ]

        # Pagination
        total_count = len(all_samples)
        start_idx = (request.page - 1) * request.page_size
        end_idx = start_idx + request.page_size
        samples = all_samples[start_idx:end_idx]

        return DatasetSamplesResponse(
            samples=samples,
            total_count=total_count,
            page=request.page,
            page_size=request.page_size,
            has_next=end_idx < total_count,
            has_prev=request.page > 1
        )

    def get_dataset_stats(self, dataset_name: str) -> DatasetStatsResponse:
        dataset_config = self.config.get_dataset_config(dataset_name)
        if not dataset_config:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        samples = self._load_dataset_samples(dataset_name, dataset_config)
        total_samples = len(samples)

        # Calculate category distribution
        categories = {}
        if dataset_config.categories:
            for sample in samples:
                category = sample.category or "unknown"
                categories[category] = categories.get(category, 0) + 1

        return DatasetStatsResponse(
            dataset_name=dataset_name,
            total_samples=total_samples,
            categories=categories if categories else None
        )

    def _count_dataset_samples(self, dataset_name: str, dataset_path: str) -> int:
        try:
            samples = self._load_dataset_samples(dataset_name,
                                               self.config.get_dataset_config(dataset_name))
            return len(samples)
        except Exception:
            return 0

    def _load_dataset_samples(self, dataset_name: str, dataset_config) -> List[DatasetSample]:
        dataset_type = dataset_config.type
        dataset_path = Path(dataset_config.path)

        if dataset_type == "mme":
            return self._load_mme_samples(dataset_path)
        elif dataset_type == "vqa":
            return self._load_vqa_samples(dataset_path)
        else:
            # Generic loader for other dataset types
            return self._load_generic_samples(dataset_path)

    def _load_mme_samples(self, dataset_path: Path) -> List[DatasetSample]:
        samples = []

        # MME dataset structure: MME_Benchmark/[category]/images/ and questions
        for category_dir in dataset_path.iterdir():
            if not category_dir.is_dir():
                continue

            category_name = category_dir.name
            images_dir = category_dir / "images"
            questions_file = category_dir / "questions_answers_YN" / f"{category_name}.txt"

            if not images_dir.exists() or not questions_file.exists():
                continue

            # Load questions and answers
            qa_data = {}
            try:
                with open(questions_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) >= 3:
                            image_name = parts[0]
                            question = parts[1]
                            answer = parts[2]
                            qa_data[image_name] = {"question": question, "answer": answer}
            except Exception:
                continue

            # Create samples for each image
            for image_file in images_dir.glob("*"):
                if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_name = image_file.name
                    qa_info = qa_data.get(image_name, {})

                    sample = DatasetSample(
                        id=f"{category_name}_{image_name}",
                        image_name=image_name,
                        image_path=str(image_file),
                        category=category_name,
                        question=qa_info.get("question"),
                        answer=qa_info.get("answer"),
                        metadata={"dataset_type": "mme"}
                    )
                    samples.append(sample)

        return samples

    def _load_vqa_samples(self, dataset_path: Path) -> List[DatasetSample]:
        # Placeholder for VQA dataset loader
        return []

    def _load_generic_samples(self, dataset_path: Path) -> List[DatasetSample]:
        samples = []

        # Generic loader: look for images and try to find associated JSON metadata
        for image_file in dataset_path.rglob("*"):
            if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                sample = DatasetSample(
                    id=str(image_file.relative_to(dataset_path)),
                    image_name=image_file.name,
                    image_path=str(image_file),
                    metadata={"dataset_type": "generic"}
                )
                samples.append(sample)

        return samples

    def get_image(self, dataset_name: str, sample_id: str) -> Optional[str]:
        """Get image path for a specific sample"""
        dataset_config = self.config.get_dataset_config(dataset_name)
        if not dataset_config:
            return None

        samples = self._load_dataset_samples(dataset_name, dataset_config)
        for sample in samples:
            if sample.id == sample_id:
                return sample.image_path
        return None