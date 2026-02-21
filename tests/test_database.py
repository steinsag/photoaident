from sqlalchemy.orm import Session

from photoaident.db.database import (
    get_engine,
    get_session_factory,
    Image,
    ImageMetadata,
    ImageTag,
    Face,
    Person,
    EmbeddingCluster,
    Suggestion,
    TakenAtSource,
    TagSource,
    FaceState,
    SuggestionState,
)


def test_create_image_with_metadata(db_session):
    img = Image(
        file_path="/path/to/image.jpg",
        file_hash="hash123",
        file_size=1024,
    )
    db_session.add(img)
    db_session.commit()

    meta = ImageMetadata(
        image_id=img.id,
        taken_at_source=TakenAtSource.EXIF,
        width=1920,
        height=1080,
    )
    db_session.add(meta)
    db_session.commit()

    assert img.id is not None
    assert img.metadata_rel is not None
    assert img.metadata_rel.width == 1920
    assert img.metadata_rel.taken_at_source == TakenAtSource.EXIF


def test_create_person_and_face(db_session):
    person = Person(name="Alice")
    db_session.add(person)
    db_session.commit()

    img = Image(file_path="img1.jpg", file_hash="h1", file_size=100)
    db_session.add(img)
    db_session.commit()

    face = Face(
        image_id=img.id,
        faiss_id=0,
        bbox_x=10,
        bbox_y=10,
        bbox_w=50,
        bbox_h=50,
        detection_confidence=0.99,
        person_id=person.id,
        state=FaceState.IDENTIFIED,
        model_version="v1",
    )
    db_session.add(face)
    db_session.commit()

    assert face.person is not None
    assert face.person.name == "Alice"
    assert len(person.faces) == 1
    assert person.faces[0].bbox_x == 10


def test_image_tags(db_session):
    img = Image(file_path="img1.jpg", file_hash="h1", file_size=100)
    db_session.add(img)
    db_session.commit()

    tag = ImageTag(
        image_id=img.id,
        tag_key="scene:beach",
        tag_value="0.95",
        tag_source=TagSource.MODEL,
        model_name="scenery-v1",
    )
    db_session.add(tag)
    db_session.commit()

    assert len(img.tags) == 1
    assert img.tags[0].tag_key == "scene:beach"


def test_embedding_clusters_and_suggestions(db_session):
    person = Person(name="Bob")
    db_session.add(person)
    db_session.commit()

    cluster = EmbeddingCluster(person_id=person.id, label="childhood")
    db_session.add(cluster)
    db_session.commit()

    img = Image(file_path="img2.jpg", file_hash="h2", file_size=100)
    db_session.add(img)
    db_session.commit()

    face = Face(
        image_id=img.id,
        faiss_id=1,
        bbox_x=0,
        bbox_y=0,
        bbox_w=10,
        bbox_h=10,
        detection_confidence=0.8,
        state=FaceState.UNIDENTIFIED,
        model_version="v1",
    )
    db_session.add(face)
    db_session.commit()

    suggestion = Suggestion(
        face_id=face.id,
        person_id=person.id,
        cluster_id=cluster.id,
        similarity_score=0.85,
        state=SuggestionState.PENDING,
    )
    db_session.add(suggestion)
    db_session.commit()

    assert len(person.clusters) == 1
    assert len(face.suggestions) == 1
    assert face.suggestions[0].person.name == "Bob"
    assert face.suggestions[0].cluster.label == "childhood"


def test_get_engine_default():
    engine = get_engine()
    assert str(engine.url).startswith("sqlite:///")


def test_get_session_factory(db_engine):
    factory = get_session_factory(db_engine)
    session = factory()
    assert isinstance(session, Session)
    session.close()
