import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

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

    property var selectedImagesById: ({})
    property var exportQueueById: ({})
    property bool selectionMode: false

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
        property string liveThumbUrl: thumbUrl
        onThumbUrlChanged: liveThumbUrl = thumbUrl
        property int pinnedElementId: 0
        property int pinnedImageId: 0

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
        height: 84
        radius: 6
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
            function onThumbnailUpdated(updatedElementId, updatedUrl) {
                if (updatedElementId === elementId) {
                    liveThumbUrl = updatedUrl
                }
            }
        }

        RowLayout {
            anchors.fill: parent
            anchors.margins: 8
            Item {
                Layout.preferredWidth: 96
                Layout.fillHeight: true
                clip: true
                Rectangle {
                    anchors.fill: parent
                    radius: 4
                    visible: liveThumbUrl.length === 0
                    color: appTheme.bgCanvasColor
                }
                BusyIndicator {
                    anchors.centerIn: parent
                    width: 28
                    height: 28
                    visible: liveThumbUrl.length === 0
                    running: visible
                }
                Image {
                    anchors.fill: parent
                    source: liveThumbUrl
                    visible: liveThumbUrl.length > 0
                    asynchronous: true
                    fillMode: Image.PreserveAspectFit
                }
            }
            ColumnLayout {
                Layout.fillWidth: true
                Label { Layout.fillWidth: true; text: fileName; color: root.rowText; font.family: appTheme.dataFontFamily; font.pixelSize: 13; elide: Text.ElideRight }
                Label { Layout.fillWidth: true; text: cameraModel + " | " + extension + " | ISO " + iso + " | f/" + aperture + " | " + focalLength + "mm"; color: root.rowMuted; font.family: appTheme.dataFontFamily; font.pixelSize: 11; elide: Text.ElideRight }
                Label { Layout.fillWidth: true; text: captureDate + " | tags: " + tags; color: root.rowMuted; font.family: appTheme.dataFontFamily; font.pixelSize: 10; elide: Text.ElideRight }
            }
            Label {
                text: rating + "/5"
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
                if (root.selectionMode) {
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
