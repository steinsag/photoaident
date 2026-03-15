"""Tests for cluster mean embedding persistence."""

import numpy as np
import pytest
from sqlalchemy import create_engine

from photoaident.db.cluster_means import (
    backfill_cluster_means,
    deserialize_embedding,
    recompute_cluster_mean,
    serialize_embedding,
)
from photoaident.db.database import (
    EmbeddingCluster,
    Face,
    FaceState,
    Image,
    Person,
    get_session_factory,
)
from photoaident.db.migrate import apply_migrations
from photoaident.db.vector_store import VectorStore


@pytest.fixture
def cluster_db(tmp_path):
    """Fresh per-test SQLite DB with migrations applied."""
    db_path = tmp_path / "cluster.db"
    apply_migrations(f"sqlite:///{db_path}")
    engine = create_engine(f"sqlite:///{db_path}")
    return get_session_factory(engine)


@pytest.fixture
def vs():
    """Fresh FAISS VectorStore."""
    return VectorStore()


# ── serialize / deserialize roundtrip ─────────────────────────────────────


def test_serialize_deserialize_roundtrip():
    """Serializing then deserializing reproduces the original embedding."""
    emb = np.random.default_rng(0).random(512).astype(np.float32)
    blob = serialize_embedding(emb)
    recovered = deserialize_embedding(blob)
    np.testing.assert_array_equal(emb, recovered)


def test_serialize_produces_correct_size():
    """512 float32 values = 2048 bytes."""
    emb = np.zeros(512, dtype=np.float32)
    assert len(serialize_embedding(emb)) == 512 * 4


# ── recompute_cluster_mean ────────────────────────────────────────────────


def test_recompute_with_no_faces(cluster_db, vs):
    """Cluster with no identified faces → mean_embedding is NULL."""
    with cluster_db() as session:
        person = Person(name="Empty")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id, label="adult", age_group="adult"
        )
        session.add(cluster)
        session.commit()
        cluster_id = cluster.id

    recompute_cluster_mean(cluster_id, cluster_db, vs)

    with cluster_db() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        assert cluster is not None
        assert cluster.mean_embedding is None


def test_recompute_with_one_face(cluster_db, vs):
    """Single face → mean equals that face's normalized embedding."""
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    faiss_id = vs.add(emb)

    with cluster_db() as session:
        person = Person(name="Solo")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id, label="adult", age_group="adult"
        )
        session.add(cluster)
        session.flush()
        cluster_id = cluster.id
        img = Image(file_path="/solo.jpg", file_size=100)
        session.add(img)
        session.flush()
        face = Face(
            image_id=img.id,
            faiss_id=faiss_id,
            bbox_x=0,
            bbox_y=0,
            bbox_w=10,
            bbox_h=10,
            detection_confidence=0.9,
            model_version="v1",
            state=FaceState.IDENTIFIED,
            person_id=person.id,
            cluster_id=cluster_id,
        )
        session.add(face)
        session.commit()

    recompute_cluster_mean(cluster_id, cluster_db, vs)

    with cluster_db() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        assert cluster is not None
        assert cluster.mean_embedding is not None
        mean = deserialize_embedding(cluster.mean_embedding)
        # Single face → mean equals the face's embedding
        np.testing.assert_allclose(mean, emb, atol=1e-6)


def test_recompute_with_multiple_faces(cluster_db, vs):
    """Two faces → mean is the normalized average."""
    emb1 = np.zeros(512, dtype=np.float32)
    emb1[0] = 1.0
    emb2 = np.zeros(512, dtype=np.float32)
    emb2[1] = 1.0
    fid1 = vs.add(emb1)
    fid2 = vs.add(emb2)

    with cluster_db() as session:
        person = Person(name="Multi")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id, label="adult", age_group="adult"
        )
        session.add(cluster)
        session.flush()
        cluster_id = cluster.id
        for fid in [fid1, fid2]:
            img = Image(file_path=f"/multi_{fid}.jpg", file_size=100)
            session.add(img)
            session.flush()
            session.add(
                Face(
                    image_id=img.id,
                    faiss_id=fid,
                    bbox_x=0,
                    bbox_y=0,
                    bbox_w=10,
                    bbox_h=10,
                    detection_confidence=0.9,
                    model_version="v1",
                    state=FaceState.IDENTIFIED,
                    person_id=person.id,
                    cluster_id=cluster_id,
                )
            )
        session.commit()

    recompute_cluster_mean(cluster_id, cluster_db, vs)

    with cluster_db() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        assert cluster is not None
        mean = deserialize_embedding(cluster.mean_embedding)
        # Mean of [1,0,...] and [0,1,...] normalized
        expected = np.zeros(512, dtype=np.float32)
        expected[0] = 0.5
        expected[1] = 0.5
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(mean, expected, atol=1e-6)


def test_recompute_stale_faiss_id_skipped(cluster_db, vs):
    """Face with faiss_id not in VectorStore is skipped gracefully."""
    with cluster_db() as session:
        person = Person(name="Stale")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id, label="adult", age_group="adult"
        )
        session.add(cluster)
        session.flush()
        cluster_id = cluster.id
        img = Image(file_path="/stale.jpg", file_size=100)
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=999,  # not in vector store
                bbox_x=0,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                detection_confidence=0.9,
                model_version="v1",
                state=FaceState.IDENTIFIED,
                person_id=person.id,
                cluster_id=cluster_id,
            )
        )
        session.commit()

    recompute_cluster_mean(cluster_id, cluster_db, vs)

    with cluster_db() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        assert cluster is not None
        assert cluster.mean_embedding is None


def test_recompute_nonexistent_cluster(cluster_db, vs):
    """Recomputing a non-existent cluster ID is a no-op."""
    recompute_cluster_mean(99999, cluster_db, vs)  # must not crash


# ── backfill_cluster_means ────────────────────────────────────────────────


def test_backfill_populates_null_means(cluster_db, vs):
    """backfill fills clusters that have NULL mean_embedding."""
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    faiss_id = vs.add(emb)

    with cluster_db() as session:
        person = Person(name="Backfill")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id, label="adult", age_group="adult"
        )
        session.add(cluster)
        session.flush()
        cluster_id = cluster.id
        img = Image(file_path="/backfill.jpg", file_size=100)
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=faiss_id,
                bbox_x=0,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                detection_confidence=0.9,
                model_version="v1",
                state=FaceState.IDENTIFIED,
                person_id=person.id,
                cluster_id=cluster_id,
            )
        )
        session.commit()

    count = backfill_cluster_means(cluster_db, vs)
    assert count == 1

    with cluster_db() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        assert cluster is not None
        assert cluster.mean_embedding is not None


def test_backfill_skips_already_computed(cluster_db, vs):
    """backfill with force=False skips clusters that already have a mean."""
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    faiss_id = vs.add(emb)

    with cluster_db() as session:
        person = Person(name="Already")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id,
            label="adult",
            age_group="adult",
            mean_embedding=serialize_embedding(emb),
        )
        session.add(cluster)
        session.flush()
        img = Image(file_path="/already.jpg", file_size=100)
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=faiss_id,
                bbox_x=0,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                detection_confidence=0.9,
                model_version="v1",
                state=FaceState.IDENTIFIED,
                person_id=person.id,
                cluster_id=cluster.id,
            )
        )
        session.commit()

    count = backfill_cluster_means(cluster_db, vs, force=False)
    assert count == 0


def test_backfill_force_recomputes_all(cluster_db, vs):
    """backfill with force=True recomputes even clusters with existing means."""
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    faiss_id = vs.add(emb)

    with cluster_db() as session:
        person = Person(name="Force")
        session.add(person)
        session.flush()
        cluster = EmbeddingCluster(
            person_id=person.id,
            label="adult",
            age_group="adult",
            mean_embedding=b"dummy",
        )
        session.add(cluster)
        session.flush()
        cluster_id = cluster.id
        img = Image(file_path="/force.jpg", file_size=100)
        session.add(img)
        session.flush()
        session.add(
            Face(
                image_id=img.id,
                faiss_id=faiss_id,
                bbox_x=0,
                bbox_y=0,
                bbox_w=10,
                bbox_h=10,
                detection_confidence=0.9,
                model_version="v1",
                state=FaceState.IDENTIFIED,
                person_id=person.id,
                cluster_id=cluster_id,
            )
        )
        session.commit()

    count = backfill_cluster_means(cluster_db, vs, force=True)
    assert count == 1

    with cluster_db() as session:
        cluster = session.get(EmbeddingCluster, cluster_id)
        assert cluster is not None
        mean = deserialize_embedding(cluster.mean_embedding)
        np.testing.assert_allclose(mean, emb, atol=1e-6)


def test_backfill_empty_db(cluster_db, vs):
    """backfill on empty DB is a no-op."""
    count = backfill_cluster_means(cluster_db, vs)
    assert count == 0
