import QtQuick
import QtQuick.Controls
import QtLocation
import QtPositioning

Item {
    id: root
    anchors.fill: parent

    // Properties for initial view (used when no bbox is provided)
    property double initialLat: 50.0
    property double initialLon: 10.0
    property int initialZoom: 5

    // Pending bbox: Python sets these via setProperty(), then sets pendingBbox=true
    // to trigger the map view to fit.  Using setProperty() (a reliable C++ API)
    // avoids the ambiguity of calling QML JavaScript functions across the boundary.
    property bool   pendingBbox:      false
    property double pendingBboxSouth: 0
    property double pendingBboxWest:  0
    property double pendingBboxNorth: 0
    property double pendingBboxEast:  0

    // Trigger immediately when Python sets pendingBbox = true.
    onPendingBboxChanged: { if (pendingBbox) _applyPendingBboxIfReady() }

    // Apply the pending bbox once the map is sized.  Safe to call repeatedly —
    // clears the flag on the first successful application.
    function _applyPendingBboxIfReady() {
        if (!pendingBbox || map.width <= 0 || map.height <= 0) return
        pendingBbox = false

        var south = pendingBboxSouth
        var west  = pendingBboxWest
        var north = pendingBboxNorth
        var east  = pendingBboxEast

        // --- Longitude (linear in WebMercator — exact) ---
        var lonSpan, centerLon
        if (east < west) {  // crosses antimeridian
            lonSpan = 360.0 - (west - east)
            var rawCenterLon = west + lonSpan / 2.0
            centerLon = rawCenterLon > 180.0 ? rawCenterLon - 360.0 : rawCenterLon
        } else {
            lonSpan = east - west
            centerLon = (west + east) / 2
        }

        // --- Latitude via exact WebMercator (Gudermannian) ---
        //
        // updateBbox uses map.toCoordinate which applies the standard WebMercator
        // formula: mercY(lat) = ln(tan(lat_rad/2 + π/4)).
        //
        // Inverting the same formula here makes the round-trip exact:
        //   • zLat solves  height*0.7*2π / (256*2^z) = mercSpan
        //   • centerLat is the Mercator midpoint, not the arithmetic degree mean
        //     (they differ due to Mercator's non-linear y-scaling)
        var northRad = north * Math.PI / 180
        var southRad = south * Math.PI / 180
        var mercN = Math.log(Math.tan(northRad / 2 + Math.PI / 4))
        var mercS = Math.log(Math.tan(southRad / 2 + Math.PI / 4))
        var mercSpan = mercN - mercS  // positive: north > south

        // Mercator-space midpoint → actual map centre latitude
        var centerLat = Math.atan(Math.sinh((mercN + mercS) / 2)) * 180 / Math.PI

        // --- Zoom (float — no rounding to preserve the exact original level) ---
        //
        // At zoom z: inner 70% rect spans  width*0.7*360/(256*2^z)  lon degrees
        //                              and  height*0.7*2π/(256*2^z)  Mercator units.
        // When the bbox was produced by this dialog both equations yield z exactly,
        // so zLon == zLat == original zoom and min(zLon, zLat) is stable.
        var zLon = lonSpan  > 0 ? Math.log2(map.width  * 0.7 * 360         / (256 * lonSpan))  : 20
        var zLat = mercSpan > 0 ? Math.log2(map.height * 0.7 * 2 * Math.PI / (256 * mercSpan)) : 20
        var zoom = Math.max(2, Math.min(15, Math.min(zLon, zLat)))

        map.center    = QtPositioning.coordinate(centerLat, centerLon)
        map.zoomLevel = zoom
    }

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
        onWidthChanged:  { updateBbox(); root._applyPendingBboxIfReady() }
        onHeightChanged: { updateBbox(); root._applyPendingBboxIfReady() }

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
