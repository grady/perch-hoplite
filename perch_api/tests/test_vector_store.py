"""Tests for Qdrant persistence and asynchronous vector writes."""

from datetime import datetime, timezone
import unittest
from unittest import mock

import httpx
import numpy as np

from perch_api.embedding import EmbeddedWindow
from perch_api.storage import S3ObjectRef
from perch_api.vector_store import QdrantStore, VectorWriter


class QdrantStoreTest(unittest.TestCase):
  def setUp(self):
    self.client = mock.MagicMock()
    self.store = QdrantStore(
        url="http://qdrant.test",
        collection="embeddings",
        client=self.client,
    )
    self.ref = S3ObjectRef("audio", "bird.wav", etag="etag")

  def test_has_vectors_returns_false_when_collection_is_missing(self):
    self.client.collection_exists.return_value = False

    self.assertFalse(self.store.has_vectors(self.ref, "perch_v2"))
    self.client.scroll.assert_not_called()

  def test_has_vectors_queries_complete_vectors_with_identity(self):
    self.client.collection_exists.return_value = True
    self.client.scroll.return_value = ([mock.sentinel.record], None)

    self.assertTrue(self.store.has_vectors(self.ref, "perch_v2"))

    request = self.client.scroll.call_args.kwargs
    self.assertEqual(request["collection_name"], "embeddings")
    self.assertEqual(request["limit"], 1)
    conditions = request["scroll_filter"].must
    self.assertEqual(
        [condition.key for condition in conditions],
        ["source", "model", "etag", "complete"],
    )

  def test_upsert_batches_points_as_complete_without_payload_overwrite(self):
    self.client.collection_exists.return_value = False
    windows = [
        EmbeddedWindow(
            vector=np.array([1.0, 2.0]),
            start_s=0.0,
            end_s=5.0,
            frame_index=0,
            channel_index=0,
            start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_time=datetime(2024, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
        ),
        EmbeddedWindow(
            vector=np.array([3.0, 4.0]),
            start_s=5.0,
            end_s=10.0,
            frame_index=1,
            channel_index=0,
        ),
    ]

    count = self.store.upsert(self.ref, "perch_v2", windows, batch_size=1)

    self.assertEqual(count, 2)
    self.client.create_collection.assert_called_once()
    self.assertEqual(self.client.upsert.call_count, 2)
    self.client.set_payload.assert_not_called()
    first_point = self.client.upsert.call_args_list[0].kwargs["points"][0]
    self.assertEqual(first_point.payload["complete"], True)
    self.assertEqual(first_point.payload["start_time"], "2024-01-01T00:00:00+00:00")
    self.assertEqual(first_point.payload["end_time"], "2024-01-01T00:00:05+00:00")

  def test_upsert_empty_windows_does_not_touch_client(self):
    self.assertEqual(self.store.upsert(self.ref, "perch_v2", []), 0)
    self.client.collection_exists.assert_not_called()
    self.client.upsert.assert_not_called()

  @mock.patch("perch_api.vector_store.time.sleep")
  def test_request_retries_transport_errors_with_exponential_backoff(self, sleep):
    operation = mock.Mock(
        side_effect=[httpx.TransportError("temporary"), "ok"]
    )

    self.assertEqual(self.store._request(operation), "ok")

    sleep.assert_called_once_with(0.5)
    self.assertEqual(operation.call_count, 2)

  def test_request_reraises_after_retry_limit(self):
    store = QdrantStore(
        url="http://qdrant.test", collection="embeddings", client=self.client, retries=0
    )
    operation = mock.Mock(side_effect=httpx.TransportError("down"))

    with self.assertRaises(httpx.TransportError):
      store._request(operation)

  @mock.patch("perch_api.vector_store.time.sleep")
  def test_request_retries_wrapped_transport_errors(self, sleep):
    wrapped = RuntimeError("qdrant response handling failed")
    wrapped.__cause__ = httpx.RemoteProtocolError("server disconnected")
    operation = mock.Mock(side_effect=[wrapped, "ok"])

    self.assertEqual(self.store._request(operation), "ok")

    sleep.assert_called_once_with(0.5)
    self.assertEqual(operation.call_count, 2)

  def test_request_does_not_retry_non_transport_errors(self):
    operation = mock.Mock(side_effect=ValueError("invalid request"))

    with self.assertRaisesRegex(ValueError, "invalid request"):
      self.store._request(operation)

    operation.assert_called_once_with()

  def test_identity_conditions_prefer_version_id_over_etag(self):
    ref = S3ObjectRef("audio", "bird.wav", version_id="v1", etag="etag")

    conditions = QdrantStore._identity_conditions(ref, "perch_v2")

    self.assertEqual([condition.key for condition in conditions], ["source", "model", "version_id"])


class VectorWriterTest(unittest.TestCase):
  def test_close_drains_queue_and_calls_callback(self):
    store = mock.MagicMock()
    store.upsert.return_value = 2
    callback = mock.Mock()
    writer = VectorWriter(store, on_write=callback)
    ref = S3ObjectRef("audio", "bird.wav")

    writer.submit(ref, "perch_v2", [])
    writer.close()

    store.upsert.assert_called_once_with(ref, "perch_v2", (), batch_size=256)
    callback.assert_called_once_with(ref, 2)

  def test_close_propagates_worker_error(self):
    store = mock.MagicMock()
    store.upsert.side_effect = RuntimeError("write failed")
    callback = mock.Mock()
    writer = VectorWriter(store, on_error=callback)
    ref = S3ObjectRef("audio", "bird.wav")

    writer.submit(ref, "perch_v2", [])

    with self.assertRaisesRegex(RuntimeError, "write failed"):
      writer.close()
    callback.assert_called_once()
    self.assertEqual(callback.call_args.args[0], ref)
    self.assertIsInstance(callback.call_args.args[1], RuntimeError)

  def test_abort_discards_queued_writes(self):
    store = mock.MagicMock()
    writer = VectorWriter(store)
    writer.submit(S3ObjectRef("audio", "bird.wav"), "perch_v2", [])
    writer.abort()

    self.assertTrue(writer._stop.is_set())


if __name__ == "__main__":
  unittest.main()
