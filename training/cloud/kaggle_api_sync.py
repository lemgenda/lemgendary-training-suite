"""Kaggle Models API synchronization service.

Interacts directly with the Kaggle Models REST API via the kaggle package to create or update
the model-level metadata and model-instance-level variation metadata (overview, usage,
license, etc.).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from training.cloud.credentials import resolve_kaggle_credentials
from training.cloud.errors import KAGGLE_API_ERRORS
from training.cloud.model_metadata import ModelMetadata
from training.cloud.variations import VARIATIONS

logger = logging.getLogger("lemtrain.cloud.kaggle_api_sync")


class KaggleApiSyncService:
    """Synchronizes model and variation metadata with Kaggle Models Hub via the official API."""

    def __init__(self, username: str | None = None, key: str | None = None) -> None:
        user, resolved_key = resolve_kaggle_credentials(override_user=username, override_key=key)
        self.username = user
        self.key = resolved_key
        self._api: KaggleApi | None = None

    def _get_api(self) -> KaggleApi:
        """Authenticate and return KaggleApi client."""
        # Ensure official Kaggle API client uses active Bearer API token
        if self.key and self.key.startswith("KGAT_"):
            os.environ["KAGGLE_API_TOKEN"] = self.key
        if self._api is None:
            api = KaggleApi()
            api.authenticate()
            self._api = api
        return self._api

    def sync_metadata(self, metadata: ModelMetadata) -> dict[str, Any]:
        """Apply model-level and instance-level metadata to Kaggle via KaggleApi."""
        api = self._get_api()
        model_slug = metadata.model.slug
        owner = metadata.model.owner
        full_model_ref = f"{owner}/{model_slug}"
        results: dict[str, Any] = {
            "model": None,
            "variations": {},
            "errors": [],
        }

        # 1. Model-level metadata sync (update first, fallback to create)
        with tempfile.TemporaryDirectory(prefix="kaggle_model_meta_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta_json = tmp_path / "model-metadata.json"
            meta_json.write_text(
                json.dumps(metadata.kaggle_model_payload(), indent=2), encoding="utf-8"
            )

            try:
                logger.info("Updating model-level metadata for '%s'...", full_model_ref)
                api.model_update(str(tmp_path))
                results["model"] = "updated"
                logger.info("Model-level metadata updated for %s", full_model_ref)
            except KAGGLE_API_ERRORS as update_err:
                logger.info(
                    "model_update failed for '%s' (%s). Attempting model_create_new...",
                    full_model_ref,
                    update_err,
                )
                try:
                    create_res = api.model_create_new(str(tmp_path))
                    results["model"] = "created"
                    logger.info(
                        "Model created on Kaggle: %s", getattr(create_res, "url", full_model_ref)
                    )
                except KAGGLE_API_ERRORS as create_err:
                    err_msg = (
                        f"Failed to create/update model {full_model_ref}: "
                        f"{update_err} | {create_err}"
                    )
                    logger.error(err_msg)
                    results["errors"].append(err_msg)

        # 2. Instance-level variation metadata sync (create or update overview/usage/license/etc)
        for variation in VARIATIONS:
            var_key = variation.key
            if var_key not in metadata.variations:
                continue
            instance_ref = f"{full_model_ref}/{variation.framework}/{variation.instance_slug}"

            with tempfile.TemporaryDirectory(
                prefix=f"kaggle_inst_meta_{variation.instance_slug}_"
            ) as tmp_dir:
                tmp_path = Path(tmp_dir)
                inst_json = tmp_path / "model-instance-metadata.json"
                inst_json.write_text(
                    json.dumps(metadata.kaggle_instance_payload(var_key), indent=2),
                    encoding="utf-8",
                )

                try:
                    # Check if instance exists
                    api.model_instance_get(instance_ref)
                    logger.info(
                        "Variation '%s' exists. Updating instance metadata...", instance_ref
                    )
                    api.model_instance_update(str(tmp_path))
                    results["variations"][var_key] = "updated"
                    logger.info("Instance metadata updated for %s", instance_ref)
                except KAGGLE_API_ERRORS as get_inst_err:
                    logger.info(
                        (
                            "Variation '%s' not found (%s). It will be registered on upload or "
                            "created."
                        ),
                        instance_ref,
                        get_inst_err,
                    )
                    try:
                        # Attempt model_instance_update anyway in case get errored
                        api.model_instance_update(str(tmp_path))
                        results["variations"][var_key] = "updated"
                    except KAGGLE_API_ERRORS as upd_err:
                        # Variation instances are formally created when binaries are
                        # uploaded via kagglehub / instance create
                        logger.info(
                            "Instance metadata update deferred until binary presence for %s (%s)",
                            instance_ref,
                            upd_err,
                        )
                        results["variations"][var_key] = "deferred"

        # 3. Synchronize default collaborators (editors) via Kaggle IAM API
        default_editors = ["lemgenda", "shyning123", "kuznetsovr"]
        try:
            from kagglesdk.security.types.iam_service import (
                GetIamPolicyRequest,
                IamPolicyBinding,
                KaggleResourceId,
                Principal,
                SetIamPolicyRequest,
                UserPrincipal,
            )

            resource_type_enum = KaggleResourceId.__init__.__globals__["KaggleResourceType"]
            canonical_role_enum = IamPolicyBinding.__init__.__globals__["CanonicalRole"]

            m_obj = api.model_get(full_model_ref)
            if hasattr(m_obj, "id") and m_obj.id:
                with api.build_kaggle_client() as kaggle_client:
                    iam_svc = kaggle_client.security.iam_client
                    g_req = GetIamPolicyRequest()
                    res_id = KaggleResourceId()
                    res_id.type = resource_type_enum.KAGGLE_RESOURCE_TYPE_MODELS
                    res_id.id = m_obj.id
                    g_req.resource_id = res_id
                    current_policy = iam_svc.get_iam_policy(g_req)

                    principals: list[Principal | None] = []
                    for uname in default_editors:
                        p = Principal()
                        u = UserPrincipal()
                        u.user_name = uname
                        p.user = u
                        principals.append(p)

                    editor_binding: IamPolicyBinding | None = None
                    existing_bindings: list[IamPolicyBinding | None] = (
                        list(current_policy.bindings) if current_policy.bindings is not None else []
                    )
                    for b in existing_bindings:
                        if b is not None and b.role == canonical_role_enum.CANONICAL_ROLE_EDITOR:
                            editor_binding = b
                            break
                    if not editor_binding:
                        editor_binding = IamPolicyBinding()
                        editor_binding.role = canonical_role_enum.CANONICAL_ROLE_EDITOR
                        existing_bindings.append(editor_binding)

                    current_policy.bindings = existing_bindings
                    editor_binding.members = principals

                    s_req = SetIamPolicyRequest()
                    s_req.resource_id = res_id
                    s_req.policy = current_policy
                    iam_svc.set_iam_policy(s_req)
                    logger.info(
                        "Synchronized collaborators for %s: %s", full_model_ref, default_editors
                    )
                    results["collaborators"] = default_editors
        except (*KAGGLE_API_ERRORS, ImportError, AttributeError) as iam_err:
            logger.warning(
                "Collaborator synchronization skipped or failed for %s: %s", full_model_ref, iam_err
            )
            results["collaborators_error"] = str(iam_err)

        return results
