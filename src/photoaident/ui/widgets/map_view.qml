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

    Plugin {
        id: mapPlugin
        name: "osm"
        PluginParameter { name: "osm.useragent"; value: root.userAgent }
        PluginParameter { name: "osm.mapping.cache.directory"; value: root.cachePath }
    }

    Map {
        id: map
        anchors.fill: parent
        plugin: mapPlugin
        center: QtPositioning.coordinate(root.initialLat, root.initialLon)
        zoomLevel: root.initialZoom

        // Explicit drag-to-pan: MapGestureArea is unreliable for mouse events
        // inside QQuickWidget on desktop, so we implement pan manually.
        MouseArea {
            anchors.fill: parent
            property real lastX: 0
            property real lastY: 0
            cursorShape: pressed ? Qt.ClosedHandCursor : Qt.OpenHandCursor

            onPressed: (mouse) => {
                lastX = mouse.x
                lastY = mouse.y
            }

            onPositionChanged: (mouse) => {
                if (pressed) {
                    var dx = mouse.x - lastX
                    var dy = mouse.y - lastY
                    map.center = map.toCoordinate(
                        Qt.point(map.width / 2 - dx, map.height / 2 - dy))
                    lastX = mouse.x
                    lastY = mouse.y
                }
            }
        }

        // Scroll-to-zoom: WheelHandler without target so it fires onWheel
        // without trying to transform the map item itself.
        WheelHandler {
            id: wheelHandler
            acceptedDevices: PointerDevice.Mouse | PointerDevice.TouchPad
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

    Component.onCompleted: map.updateBbox()
}
