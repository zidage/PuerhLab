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
    readonly property color rowBgSelected: Qt.rgba(252 / 255, 199 / 255, 4 / 255, 0.10)
    readonly property color rowBgHover: "#252525"
    readonly property color rowMuted: "#888888"
    readonly property color rowText: "#E6E6E6"
    readonly property color rowAccent: "#FCC704"

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
            ColumnLayout {
                Layout.fillWidth: true
                Label { Layout.fillWidth: true; text: fileName; color: root.rowText; font.pixelSize: 13; elide: Text.ElideRight }
                Label { Layout.fillWidth: true; text: cameraModel + " | " + extension + " | ISO " + iso + " | f/" + aperture + " | " + focalLength + "mm"; color: root.rowMuted; font.pixelSize: 11; elide: Text.ElideRight }
                Label { Layout.fillWidth: true; text: captureDate + " | tags: " + tags; color: root.rowMuted; font.pixelSize: 10; elide: Text.ElideRight }
            }
            Label {
                text: rating + "/5"
                color: root.rowText
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
            cursorShape: Qt.PointingHandCursor
            onClicked: {
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
            onDoubleClicked: albumBackend.openEditor(elementId, imageId)
        }
    }
}
