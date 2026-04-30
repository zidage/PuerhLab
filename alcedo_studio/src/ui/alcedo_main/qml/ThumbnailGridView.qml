import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

Item {
    id: root
    clip: true
    readonly property color cardBg: "transparent"
    readonly property color cardBgSelected: appTheme.selectedTintColor
    readonly property color cardBgHover: appTheme.hoverColor
    readonly property color cardMuted: appTheme.textMutedColor
    readonly property color cardText: appTheme.textColor
    readonly property color cardAccent: appTheme.accentColor
    readonly property color cardDanger: appTheme.dangerColor
    readonly property color cardDangerTint: appTheme.dangerTintColor
    readonly property int dataFontWeight: 500
    readonly property real dataLetterSpacing: -0.2

    property var selectedImagesById: ({})
    property var exportQueueById: ({})

    signal imageSelectionChanged(int elementId, int imageId, string fileName, bool selected)
    signal replaceSelection(var items)
    signal contextMenuRequested(var item, real sceneX, real sceneY)

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

    function hasMultiSelectModifier(modifiers) {
        return (modifiers & Qt.ShiftModifier) || (modifiers & Qt.ControlModifier)
    }

    function selectionItemForIndex(index) {
        if (index < 0 || index >= albumBackend.thumbnails.length) {
            return null
        }

        const row = albumBackend.thumbnails[index]
        if (!row) {
            return null
        }
        const elementId = Number(row.elementId)
        if (elementId <= 0) {
            return null
        }

        return {
            elementId: elementId,
            imageId: Number(row.imageId),
            fileName: row.fileName ? row.fileName : qsTr("(unnamed)")
        }
    }

    GridView {
        id: grid
        anchors.fill: parent
        model: albumBackend.thumbnails
        clip: true
        cacheBuffer: 0
        cellWidth: 242
        cellHeight: 240
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
        required property bool thumbLoading
        required property bool thumbMissingSource
        property string liveThumbUrl: thumbUrl
        onThumbUrlChanged: liveThumbUrl = thumbUrl
        property bool liveThumbLoading: thumbLoading
        onThumbLoadingChanged: liveThumbLoading = thumbLoading
        property bool liveThumbMissingSource: thumbMissingSource
        onThumbMissingSourceChanged: liveThumbMissingSource = thumbMissingSource
        property int pinnedElementId: 0
        property int pinnedImageId: 0
        readonly property bool thumbnailReady: liveThumbUrl.length > 0
        readonly property bool thumbnailLoadingState: liveThumbLoading
        readonly property bool thumbnailMissingState: !thumbnailReady && !thumbnailLoadingState && liveThumbMissingSource
        readonly property bool thumbnailIdleState: !thumbnailReady && !thumbnailLoadingState && !thumbnailMissingState

        function bindThumbnailLifetime() {
            if (pinnedElementId === elementId && pinnedImageId === imageId) {
                return
            }
            if (pinnedElementId !== 0 && pinnedImageId !== 0) {
                albumBackend.SetThumbnailVisible(pinnedElementId, pinnedImageId, false)
            }
            pinnedElementId = elementId
            pinnedImageId = imageId
            liveThumbUrl = thumbUrl
            liveThumbLoading = thumbLoading
            liveThumbMissingSource = thumbMissingSource
            if (pinnedElementId !== 0 && pinnedImageId !== 0) {
                albumBackend.SetThumbnailVisible(pinnedElementId, pinnedImageId, true)
            }
        }

        Component.onCompleted: bindThumbnailLifetime()
        onElementIdChanged: bindThumbnailLifetime()
        onImageIdChanged: bindThumbnailLifetime()
        Component.onDestruction: {
            if (pinnedElementId !== 0 && pinnedImageId !== 0) {
                albumBackend.SetThumbnailVisible(pinnedElementId, pinnedImageId, false)
            }
        }

            readonly property bool isSelected: root.isImageSelected(elementId)
            readonly property bool isHovered: overlay.hoveredIndex === index

        width: 230
        height: 224
        radius: appTheme.panelRadius
            color: isSelected ? root.cardBgSelected
                   : (isHovered ? root.cardBgHover : root.cardBg)
            border.width: isSelected ? 2 : 0
            border.color: root.cardAccent
        Behavior on color { ColorAnimation { duration: 120 } }
            Behavior on border.width { NumberAnimation { duration: 150 } }

        Connections {
            target: albumBackend
            ignoreUnknownSignals: true
            function onThumbnailUpdated(updatedElementId, updatedUrl, loading, missingSource) {
                if (updatedElementId === elementId) {
                    liveThumbUrl = updatedUrl
                    liveThumbLoading = loading
                    liveThumbMissingSource = missingSource
                }
            }
        }

            Item {
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 8
                height: width * 2 / 3
                clip: true

                Rectangle {
                    anchors.fill: parent
                    radius: 10
                    color: appTheme.bgBaseColor
                    border.width: 2
                    border.color: appTheme.dividerColor
                }
                BusyIndicator {
                    anchors.centerIn: parent
                    width: 28
                    height: 28
                    visible: thumbnailLoadingState
                    running: visible
                }
                Image {
                    id: thumbImage
                    anchors.centerIn: parent
                    width: parent.width - 4
                    height: parent.height - 4
                    source: liveThumbUrl
                    visible: false
                    asynchronous: true
                    fillMode: Image.PreserveAspectFit
                }
                Rectangle {
                    id: thumbMask
                    anchors.fill: thumbImage
                    radius: 8
                    visible: false
                    layer.enabled: true
                }
                MultiEffect {
                    anchors.fill: thumbImage
                    source: thumbImage
                    maskEnabled: true
                    maskSource: thumbMask
                    visible: thumbnailReady
                }
                Rectangle {
                    anchors.top: parent.top
                    anchors.right: parent.right
                    anchors.margins: 8
                    width: 18
                    height: 18
                    radius: 9
                    visible: thumbnailMissingState
                    color: root.cardDangerTint
                    border.width: 1
                    border.color: root.cardDanger
                }
                Label {
                    anchors.centerIn: parent
                    visible: thumbnailMissingState
                    text: "!"
                    color: root.cardDanger
                    font.family: appTheme.dataFontFamily
                    font.pixelSize: 30
                    font.weight: 700
                }
                HoverHandler {
                    id: thumbHover
                }
                ToolTip.visible: thumbnailMissingState && thumbHover.hovered
                ToolTip.text: qsTr("Source file was moved or deleted")
                ToolTip.delay: 150
        }

        Column {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 8
            spacing: 2
            Label {
                text: fileName
                color: root.cardText
                font.family: appTheme.dataFontFamily
                font.pixelSize: 12
                font.weight: root.dataFontWeight
                font.letterSpacing: root.dataLetterSpacing
                elide: Text.ElideRight
                width: parent.width
            }
            Label {
                text: qsTr("%1 | ISO %2 | f/%3").arg(cameraModel).arg(iso).arg(aperture)
                color: root.cardMuted
                font.family: appTheme.dataFontFamily
                font.pixelSize: 10
                font.weight: root.dataFontWeight
                font.letterSpacing: root.dataLetterSpacing
                elide: Text.ElideRight
                width: parent.width
            }
            Label {
                text: qsTr("%1 | Rating %2/5").arg(captureDate).arg(rating)
                color: root.cardMuted
                font.family: appTheme.dataFontFamily
                font.pixelSize: 10
                font.weight: root.dataFontWeight
                font.letterSpacing: root.dataLetterSpacing
                elide: Text.ElideRight
                width: parent.width
            }
        }

        }
    }

    // ── Interaction overlay ──
    MouseArea {
        id: overlay
        anchors.fill: parent
        hoverEnabled: true
        acceptedButtons: Qt.LeftButton | Qt.RightButton

        property int hoveredIndex: -1
        property point dragStart: Qt.point(0, 0)
        property point dragCurrent: Qt.point(0, 0)
        property real dragStartContentY: 0
        property bool isDragging: false
        property var preDragSelection: ({})
        property bool dragAdditive: false

        cursorShape: hoveredIndex >= 0 ? Qt.PointingHandCursor : Qt.ArrowCursor

        function gridIndexAt(viewX, viewY) {
            return grid.indexAt(viewX + grid.contentX, viewY + grid.contentY)
        }

        function dragStartContentTop() {
            return dragStart.y + dragStartContentY
        }

        function dragCurrentContentTop() {
            return dragCurrent.y + grid.contentY
        }

        function rubberBandViewportY() {
            return Math.min(dragStartContentTop(), dragCurrentContentTop()) - grid.contentY
        }

        function rubberBandViewportHeight() {
            return Math.abs(dragCurrentContentTop() - dragStartContentTop())
        }

        function applyRubberBandSelection() {
            const bandItems = collectRubberBandItems()
            if (dragAdditive) {
                const merged = Object.values(preDragSelection).concat(bandItems)
                root.replaceSelection(merged)
            } else {
                root.replaceSelection(bandItems)
            }
        }

        function collectRubberBandItems() {
            const colCount = Math.max(1, Math.floor(grid.width / grid.cellWidth))
            const totalCount = grid.count

            const bLeft   = Math.min(dragStart.x, dragCurrent.x)
            const bRight  = Math.max(dragStart.x, dragCurrent.x)
            const bTop    = Math.min(dragStartContentTop(), dragCurrentContentTop())
            const bBottom = Math.max(dragStartContentTop(), dragCurrentContentTop())

            const minCol = Math.max(0, Math.floor(bLeft / grid.cellWidth))
            const maxCol = Math.min(colCount - 1, Math.floor(bRight / grid.cellWidth))
            const minRow = Math.max(0, Math.floor(bTop / grid.cellHeight))
            const maxRow = Math.floor(bBottom / grid.cellHeight)

            const items = []
            for (let row = minRow; row <= maxRow; ++row) {
                for (let col = minCol; col <= maxCol; ++col) {
                    const idx = row * colCount + col
                    if (idx >= 0 && idx < totalCount) {
                        const item = root.selectionItemForIndex(idx)
                        if (item) {
                            items.push(item)
                        }
                    }
                }
            }
            return items
        }

        onPositionChanged: function(mouse) {
            if (pressed && (pressedButtons & Qt.LeftButton)) {
                const dx = mouse.x - dragStart.x
                const dy = mouse.y - dragStart.y
                if (!isDragging && (dx * dx + dy * dy) > 64) {
                    isDragging = true
                    dragAdditive = root.hasMultiSelectModifier(mouse.modifiers)
                    if (dragAdditive) {
                        preDragSelection = Object.assign({}, root.selectedImagesById)
                    }
                }
                if (isDragging) {
                    dragCurrent = Qt.point(mouse.x, mouse.y)
                    applyRubberBandSelection()
                }
            }
            hoveredIndex = gridIndexAt(mouse.x, mouse.y)
        }

        onPressed: function(mouse) {
            if (mouse.button === Qt.RightButton) {
                const idx = gridIndexAt(mouse.x, mouse.y)
                if (idx >= 0) {
                    const item = root.selectionItemForIndex(idx)
                    if (item) {
                        const scenePoint = overlay.mapToItem(null, mouse.x, mouse.y)
                        root.contextMenuRequested(item, scenePoint.x, scenePoint.y)
                    }
                }
                return
            }
            dragStart = Qt.point(mouse.x, mouse.y)
            dragCurrent = Qt.point(mouse.x, mouse.y)
            dragStartContentY = grid.contentY
            isDragging = false
            dragAdditive = false
            preDragSelection = ({})
        }

        onReleased: function(mouse) {
            if (mouse.button !== Qt.LeftButton) {
                isDragging = false
                return
            }
            if (!isDragging) {
                const idx = gridIndexAt(mouse.x, mouse.y)
                if (idx >= 0) {
                    const item = root.selectionItemForIndex(idx)
                    if (item) {
                        if (root.hasMultiSelectModifier(mouse.modifiers)) {
                            const next = !root.isImageSelected(item.elementId)
                            root.imageSelectionChanged(item.elementId, item.imageId, item.fileName, next)
                        } else {
                            root.replaceSelection([item])
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
                const item = root.selectionItemForIndex(idx)
                if (item) {
                    albumBackend.OpenEditor(item.elementId, item.imageId)
                }
            }
        }

        onExited: hoveredIndex = -1

        onWheel: function(wheel) {
            grid.contentY = Math.max(0, Math.min(
                grid.contentHeight - grid.height,
                grid.contentY - wheel.angleDelta.y))
            if (isDragging) {
                applyRubberBandSelection()
            }
            hoveredIndex = gridIndexAt(mouseX, mouseY)
            wheel.accepted = true
        }
    }

    // ── Rubber band visual ──
    Rectangle {
        id: rubberBand
        visible: overlay.isDragging
        x: Math.min(overlay.dragStart.x, overlay.dragCurrent.x)
        y: overlay.rubberBandViewportY()
        width: Math.abs(overlay.dragCurrent.x - overlay.dragStart.x)
        height: overlay.rubberBandViewportHeight()
        color: Qt.rgba(appTheme.toneMist.r, appTheme.toneMist.g, appTheme.toneMist.b, 0.08)
        border.width: 1
        border.color: Qt.rgba(appTheme.toneMist.r, appTheme.toneMist.g, appTheme.toneMist.b, 0.50)
        radius: 2
        z: 10
    }
}
