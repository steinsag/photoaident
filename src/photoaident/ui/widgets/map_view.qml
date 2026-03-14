import QtQuick
import QtQuick.Controls
import QtLocation
import QtPositioning

Item {
    id: root
    anchors.fill: parent

    // Properties for initial view
    property double initialLat: 50.0
    property double initialLon: 10.0
    property int initialZoom: 5

    // Parameters for OSM plugin
    property string userAgent: "PhotoAIdent/0.1"
    property string cachePath: ""

    // Bbox values updated as map moves/zooms
    property double south: 0.0
    property double west: 0.0
    property double north: 0.0
    property double east: 0.0

    function zoomIn() {
        map.zoomLevel = Math.min(map.maximumZoomLevel, map.zoomLevel + 1)
    }

    function zoomOut() {
        map.zoomLevel = Math.max(map.minimumZoomLevel, map.zoomLevel - 1)
    }

    Plugin {
        id: mapPlugin
        name: "osm"
        PluginParameter { name: "osm.useragent"; value: root.userAgent }
        PluginParameter { name: "osm.mapping.cache.directory"; value: root.cachePath }
        // Qt 6.5+ changed the default street map to Stadia Maps (API key required).
        // Use the free OpenStreetMap tile server via the "custom" map type instead.
        PluginParameter { name: "osm.mapping.custom.host"; value: "https://tile.openstreetmap.org/" }
    }

    Map {
        id: map
        anchors.fill: parent
        plugin: mapPlugin
        center: QtPositioning.coordinate(root.initialLat, root.initialLon)
        zoomLevel: root.initialZoom

        // DragHandler cooperates with other pointer handlers (unlike MouseArea)
        // so pinch and wheel events are not accidentally consumed.
        DragHandler {
            id: dragHandler
            target: null
            cursorShape: active ? Qt.ClosedHandCursor : Qt.OpenHandCursor

            property real prevX: 0
            property real prevY: 0

            onActiveChanged: {
                if (active) {
                    prevX = 0
                    prevY = 0
                }
            }

            onTranslationChanged: {
                if (!active) return
                var dx = translation.x - prevX
                var dy = translation.y - prevY
                map.center = map.toCoordinate(
                    Qt.point(map.width / 2 - dx, map.height / 2 - dy))
                prevX = translation.x
                prevY = translation.y
            }
        }

        // PinchHandler for pinch-to-zoom on touch screens and trackpads
        PinchHandler {
            id: pinchHandler
            target: null
            property real startZoom: 0

            onActiveChanged: {
                if (active) {
                    startZoom = map.zoomLevel
                }
            }

            onActiveScaleChanged: {
                if (!active) return
                map.zoomLevel = Math.max(
                    map.minimumZoomLevel,
                    Math.min(map.maximumZoomLevel,
                             startZoom + Math.log2(activeScale)))
            }
        }

        // Scroll-to-zoom: WheelHandler without target so it fires onWheel
        // without trying to transform the map item itself.
        WheelHandler {
            id: wheelHandler
            acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad | PointerDevice.TouchScreen
            onWheel: (event) => {
                map.zoomLevel = Math.max(
                    map.minimumZoomLevel,
                    Math.min(map.maximumZoomLevel,
                             map.zoomLevel + event.angleDelta.y / 120))
            }
        }

        // Update bbox whenever center or zoom changes (more reliable than
        // onVisibleRegionChanged which fires for geocircles/polygons).
        onCenterChanged: updateBbox()
        onZoomLevelChanged: updateBbox()
        onWidthChanged: updateBbox()
        onHeightChanged: updateBbox()

        function updateBbox() {
            // Use the inner 70% of the viewport as the search area.
            // Qt.point() constructs new value-type objects to avoid mutating
            // the result of fromCoordinate() in place (unreliable across versions).
            var rectW = width * 0.7
            var rectH = height * 0.7
            var centerPt = map.fromCoordinate(map.center)
            var topLeft = Qt.point(centerPt.x - rectW / 2, centerPt.y - rectH / 2)
            var bottomRight = Qt.point(centerPt.x + rectW / 2, centerPt.y + rectH / 2)
            var coordTopLeft = map.toCoordinate(topLeft)
            var coordBottomRight = map.toCoordinate(bottomRight)
            root.north = coordTopLeft.latitude
            root.south = coordBottomRight.latitude
            root.west = coordTopLeft.longitude
            root.east = coordBottomRight.longitude
        }

        // Overlay rectangle to show the search area
        Rectangle {
            id: selectionRect
            anchors.centerIn: parent
            width: parent.width * 0.7
            height: parent.height * 0.7
            color: "transparent"
            border.color: "blue"
            border.width: 2
            opacity: 0.6

            Rectangle {
                anchors.fill: parent
                color: "blue"
                opacity: 0.1
            }
        }
    }

    Component.onCompleted: {
        // Activate the custom map type so osm.mapping.custom.host is used
        // instead of the default Stadia Maps type (which requires an API key).
        for (var i = 0; i < map.supportedMapTypes.length; i++) {
            if (map.supportedMapTypes[i].style === MapType.CustomMap) {
                map.activeMapType = map.supportedMapTypes[i]
                break
            }
        }
        map.updateBbox()
    }
}
