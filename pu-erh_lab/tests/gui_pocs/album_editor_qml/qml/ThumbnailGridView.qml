import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: root
    clip: true
    readonly property color cardBg: "transparent"
    readonly property color cardBgSelected: Qt.rgba(252 / 255, 199 / 255, 4 / 255, 0.10)
    readonly property color cardBgHover: "#252525"
    readonly property color cardMuted: "#888888"
    readonly property color cardText: "#E6E6E6"
    readonly property color cardAccent: "#FCC704"

    property var selectedImagesById: ({})
    property var exportQueueById: ({})
    property bool selectionMode: false

    signal imageSelectionChanged(int elementId, int imageId, string fileName, bool selected)
    signal replaceSelection(var items)

    function keyForElement(elementId) {
        return String(Number(elementId))
    }

    function isImageSelected(elementId) {
        return Object.prototype.hasOwnProperty.call(
            selectedImagesById, keyForElement(elementId))
    }

    function isImageQueued(elementId) {
        return Object.prototype.hasOwnProperty.call(
            exportQueueById, keyForElement(elementId))
    }

    GridView {
        id: grid
        anchors.fill: parent
        model: albumBackend.thumbnails
        clip: true
        cacheBuffer: 0
        cellWidth: 242
        cellHeight: 186
        interactive: false

        delegate: Rectangle {
            id: cardDelegate
            required property int index
            required property int elementId
            required property int imageId
        required property string fileName
        required property string cameraModel
        required property int iso
        required property string aperture
        required property string captureDate
        required property int rating
        required property string accent
        required property string thumbUrl
        property string liveThumbUrl: thumbUrl
        onThumbUrlChanged: liveThumbUrl = thumbUrl
        property int pinnedElementId: 0
        property int pinnedImageId: 0

        function bindThumbnailLifetime() {
            if (pinnedElementId === elementId && pinnedImageId === imageId) {
                return
            }
            if (pinnedElementId !== 0 && pinnedImageId !== 0) {
                albumBackend.setThumbnailVisible(pinnedElementId, pinnedImageId, false)
            }
            pinnedElementId = elementId
            pinnedImageId = imageId
            liveThumbUrl = thumbUrl
            if (pinnedElementId !== 0 && pinnedImageId !== 0) {
                albumBackend.setThumbnailVisible(pinnedElementId, pinnedImageId, true)
            }
        }

        Component.onCompleted: bindThumbnailLifetime()
        onElementIdChanged: bindThumbnailLifetime()
        onImageIdChanged: bindThumbnailLifetime()
        Component.onDestruction: {
            if (pinnedElementId !== 0 && pinnedImageId !== 0) {
                albumBackend.setThumbnailVisible(pinnedElementId, pinnedImageId, false)
            }
        }

            readonly property bool isSelected: root.isImageSelected(elementId)
            readonly property bool isHovered: overlay.hoveredIndex === index

        width: 230
        height: 172
        radius: 6
            color: isSelected ? root.cardBgSelected
                   : (isHovered ? root.cardBgHover : root.cardBg)
            border.width: isSelected ? 2 : 0
            border.color: root.cardAccent
        Behavior on color { ColorAnimation { duration: 120 } }
            Behavior on border.width { NumberAnimation { duration: 150 } }

        Connections {
            target: albumBackend
            function onThumbnailUpdated(updatedElementId, updatedUrl) {
                if (updatedElementId === elementId) {
                    liveThumbUrl = updatedUrl
                }
            }
        }

            Item {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 6
            height: 96
            clip: true

                Rectangle {
                    anchors.fill: parent
                    radius: 4
                    visible: liveThumbUrl.length === 0
                    color: "#0D0D0D"
                }
                Image {
                    anchors.fill: parent
                    source: liveThumbUrl
                    visible: liveThumbUrl.length > 0
                    asynchronous: true
                    fillMode: Image.PreserveAspectFit
                }
        }

        Column {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 6
            spacing: 1
            Label { text: fileName; color: root.cardText; font.pixelSize: 12; elide: Text.ElideRight; width: parent.width }
            Label { text: cameraModel + " | ISO " + iso + " | f/" + aperture; color: root.cardMuted; font.pixelSize: 10; elide: Text.ElideRight; width: parent.width }
            Label { text: captureDate + " | rating " + rating + "/5"; color: root.cardMuted; font.pixelSize: 10; elide: Text.ElideRight; width: parent.width }
        }

        }
    }

    // ── Interaction overlay ──
    MouseArea {
        id: overlay
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton

        property int hoveredIndex: -1
        property point dragStart: Qt.point(0, 0)
        property point dragCurrent: Qt.point(0, 0)
        property bool isDragging: false
        property var preDragSelection: ({})

        cursorShape: hoveredIndex >= 0 ? Qt.PointingHandCursor : Qt.ArrowCursor

        function gridIndexAt(viewX, viewY) {
            return grid.indexAt(viewX + grid.contentX, viewY + grid.contentY)
        }

        function collectRubberBandItems() {
            const colCount = Math.max(1, Math.floor(grid.width / grid.cellWidth))
            const totalCount = grid.count

            const bLeft   = Math.min(dragStart.x, dragCurrent.x)
            const bRight  = Math.max(dragStart.x, dragCurrent.x)
            const bTop    = Math.min(dragStart.y, dragCurrent.y) + grid.contentY
            const bBottom = Math.max(dragStart.y, dragCurrent.y) + grid.contentY

            const minCol = Math.max(0, Math.floor(bLeft / grid.cellWidth))
            const maxCol = Math.min(colCount - 1, Math.floor(bRight / grid.cellWidth))
            const minRow = Math.max(0, Math.floor(bTop / grid.cellHeight))
            const maxRow = Math.floor(bBottom / grid.cellHeight)

            const items = []
            for (let row = minRow; row <= maxRow; ++row) {
                for (let col = minCol; col <= maxCol; ++col) {
                    const idx = row * colCount + col
                    if (idx >= 0 && idx < totalCount) {
                        const item = grid.itemAtIndex(idx)
                        if (item) {
                            items.push({
                                elementId: item.elementId,
                                imageId: item.imageId,
                                fileName: item.fileName
                            })
                        }
                    }
                }
            }
            return items
        }

        onPositionChanged: function(mouse) {
            if (pressed) {
                const dx = mouse.x - dragStart.x
                const dy = mouse.y - dragStart.y
                if (!isDragging && (dx * dx + dy * dy) > 64) {
                    isDragging = true
                    if (root.selectionMode) {
                        preDragSelection = Object.assign({}, root.selectedImagesById)
                    }
                }
                if (isDragging) {
                    dragCurrent = Qt.point(mouse.x, mouse.y)
                    const bandItems = collectRubberBandItems()
                    if (root.selectionMode) {
                        const merged = Object.values(preDragSelection).concat(bandItems)
                        root.replaceSelection(merged)
                    } else {
                        root.replaceSelection(bandItems)
                    }
                }
            } else {
                hoveredIndex = gridIndexAt(mouse.x, mouse.y)
            }
        }

        onPressed: function(mouse) {
            dragStart = Qt.point(mouse.x, mouse.y)
            dragCurrent = Qt.point(mouse.x, mouse.y)
            isDragging = false
        }

        onReleased: function(mouse) {
            if (!isDragging) {
                const idx = gridIndexAt(mouse.x, mouse.y)
                if (idx >= 0) {
                    const item = grid.itemAtIndex(idx)
                    if (item) {
                        if (root.selectionMode) {
                            const next = !root.isImageSelected(item.elementId)
                            root.imageSelectionChanged(item.elementId, item.imageId, item.fileName, next)
                        } else {
                            root.replaceSelection([{
                                elementId: item.elementId,
                                imageId: item.imageId,
                                fileName: item.fileName
                            }])
                        }
                    }
                } else {
                    root.replaceSelection([])
                }
            }
            isDragging = false
        }

        onDoubleClicked: function(mouse) {
            const idx = gridIndexAt(mouse.x, mouse.y)
            if (idx >= 0) {
                const item = grid.itemAtIndex(idx)
                if (item) {
                    albumBackend.openEditor(item.elementId, item.imageId)
                }
            }
        }

        onExited: hoveredIndex = -1

        onWheel: function(wheel) {
            grid.contentY = Math.max(0, Math.min(
                grid.contentHeight - grid.height,
                grid.contentY - wheel.angleDelta.y))
            hoveredIndex = gridIndexAt(mouseX, mouseY)
            wheel.accepted = true
        }
    }

    // ── Rubber band visual ──
    Rectangle {
        id: rubberBand
        visible: overlay.isDragging
        x: Math.min(overlay.dragStart.x, overlay.dragCurrent.x)
        y: Math.min(overlay.dragStart.y, overlay.dragCurrent.y)
        width: Math.abs(overlay.dragCurrent.x - overlay.dragStart.x)
        height: Math.abs(overlay.dragCurrent.y - overlay.dragStart.y)
        color: Qt.rgba(252 / 255, 199 / 255, 4 / 255, 0.08)
        border.width: 1
        border.color: Qt.rgba(252 / 255, 199 / 255, 4 / 255, 0.50)
        radius: 2
        z: 10
    }
}
