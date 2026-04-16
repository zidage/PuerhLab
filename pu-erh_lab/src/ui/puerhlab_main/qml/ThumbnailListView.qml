import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Effects

ListView {
    id: root
    model: albumBackend.thumbnails
    clip: true
    cacheBuffer: 0
    spacing: 8
    readonly property color rowBg: "transparent"
    readonly property color rowBgSelected: appTheme.selectedTintColor
    readonly property color rowBgHover: appTheme.hoverColor
    readonly property color rowMuted: appTheme.textMutedColor
    readonly property color rowText: appTheme.textColor
    readonly property color rowAccent: appTheme.accentColor
    readonly property color rowDanger: appTheme.dangerColor
    readonly property color rowDangerTint: appTheme.dangerTintColor

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

    delegate: Rectangle {
        required property int elementId
        required property int imageId
        required property string fileName
        required property string cameraModel
        required property string extension
        required property int iso
        required property string aperture
        required property string focalLength
        required property string captureDate
        required property int rating
        required property string tags
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

        width: ListView.view.width
        height: 116
        radius: appTheme.panelRadius
        color: root.isImageSelected(elementId)
              ? root.rowBgSelected
              : (rowHoverArea.containsMouse ? root.rowBgHover : root.rowBg)
        border.width: root.isImageSelected(elementId) ? 2 : 0
        border.color: root.rowAccent
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

        RowLayout {
            anchors.fill: parent
            anchors.margins: 8
            Item {
                Layout.preferredWidth: 132
                Layout.preferredHeight: 88
                Layout.alignment: Qt.AlignVCenter
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
                    anchors.margins: 6
                    width: 16
                    height: 16
                    radius: 8
                    visible: thumbnailMissingState
                    color: root.rowDangerTint
                    border.width: 1
                    border.color: root.rowDanger
                }
                Label {
                    anchors.centerIn: parent
                    visible: thumbnailMissingState
                    text: "!"
                    color: root.rowDanger
                    font.family: appTheme.dataFontFamily
                    font.pixelSize: 28
                    font.weight: 700
                }
                HoverHandler {
                    id: thumbHover
                }
                ToolTip.visible: thumbnailMissingState && thumbHover.hovered
                ToolTip.text: qsTr("Source file was moved or deleted")
                ToolTip.delay: 150
            }
            ColumnLayout {
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignVCenter
                spacing: 4
                Label { Layout.fillWidth: true; text: fileName; color: root.rowText; font.family: appTheme.dataFontFamily; font.pixelSize: 13; elide: Text.ElideRight }
                Label {
                    Layout.fillWidth: true
                    text: qsTr("%1 | %2 | ISO %3 | f/%4 | %5mm")
                        .arg(cameraModel)
                        .arg(extension)
                        .arg(iso)
                        .arg(aperture)
                        .arg(focalLength)
                    color: root.rowMuted
                    font.family: appTheme.dataFontFamily
                    font.pixelSize: 11
                    elide: Text.ElideRight
                }
                Label {
                    Layout.fillWidth: true
                    text: qsTr("%1 | Tags: %2").arg(captureDate).arg(tags)
                    color: root.rowMuted
                    font.family: appTheme.dataFontFamily
                    font.pixelSize: 10
                    elide: Text.ElideRight
                }
            }
            Label {
                text: qsTr("%1/5").arg(rating)
                color: root.rowText
                font.family: appTheme.dataFontFamily
                font.pixelSize: 12
                font.weight: 700
                horizontalAlignment: Text.AlignHCenter
                Layout.alignment: Qt.AlignVCenter
            }
        }

        MouseArea {
            id: rowHoverArea
            anchors.fill: parent
            hoverEnabled: true
            acceptedButtons: Qt.LeftButton | Qt.RightButton
            cursorShape: Qt.PointingHandCursor
            onPressed: function(mouse) {
                if (mouse.button !== Qt.RightButton) {
                    return
                }
                const scenePoint = rowHoverArea.mapToItem(null, mouse.x, mouse.y)
                root.contextMenuRequested({
                    elementId: elementId,
                    imageId: imageId,
                    fileName: fileName
                }, scenePoint.x, scenePoint.y)
            }
            onClicked: function(mouse) {
                if (mouse.button !== Qt.LeftButton) {
                    return
                }
                if (root.hasMultiSelectModifier(mouse.modifiers)) {
                    const nextSelected = !root.isImageSelected(elementId)
                    root.imageSelectionChanged(elementId, imageId, fileName, nextSelected)
                } else {
                    root.replaceSelection([{
                        elementId: elementId,
                        imageId: imageId,
                        fileName: fileName
                    }])
                }
            }
            onDoubleClicked: albumBackend.OpenEditor(elementId, imageId)
        }
    }
}
