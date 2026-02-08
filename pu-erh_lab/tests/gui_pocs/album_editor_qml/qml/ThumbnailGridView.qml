import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

GridView {
    id: root
    model: albumBackend.thumbnails
    clip: true
    cellWidth: 242
    cellHeight: 186

    property var selectedImagesById: ({})
    property var exportQueueById: ({})
    signal imageSelectionChanged(int elementId, int imageId, string fileName, bool selected)

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
        required property int iso
        required property string aperture
        required property string captureDate
        required property int rating
        required property string accent
        required property string thumbUrl
        property string liveThumbUrl: thumbUrl
        onThumbUrlChanged: liveThumbUrl = thumbUrl

        width: 230
        height: 172
        radius: 12
        color: "#4D4C51"
        border.color: root.isImageSelected(elementId)
                      ? "#92BCE1"
                      : (root.isImageQueued(elementId)
                         ? "#A1AC9B"
                         : (albumBackend.editorActive && albumBackend.editorElementId === elementId
                            ? "#E03D46"
                            : "#4D4C51"))

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
            anchors.margins: 8
            height: 92
            clip: true

            Rectangle {
                anchors.fill: parent
                radius: 9
                visible: liveThumbUrl.length === 0
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "#4D4C51" }
                    GradientStop { position: 1.0; color: "#38373C" }
                }
            }
            Image {
                anchors.fill: parent
                source: liveThumbUrl
                visible: liveThumbUrl.length > 0
                asynchronous: true
                fillMode: Image.PreserveAspectFit
            }
            CheckBox {
                anchors.left: parent.left
                anchors.top: parent.top
                anchors.margins: 6
                checked: root.isImageSelected(elementId)
                onClicked: root.imageSelectionChanged(elementId, imageId, fileName, checked)
            }
            Button {
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 6
                text: "Edit"
                onClicked: albumBackend.openEditor(elementId, imageId)
            }
        }

        Column {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            anchors.margins: 8
            spacing: 2
            Label { text: fileName; color: "#E3DFDB"; font.pixelSize: 12; elide: Text.ElideRight; width: parent.width }
            Label { text: cameraModel + " | ISO " + iso + " | f/" + aperture; color: "#7B7D7C"; font.pixelSize: 11; elide: Text.ElideRight; width: parent.width }
            Label { text: captureDate + " | rating " + rating + "/5"; color: "#7B7D7C"; font.pixelSize: 10; elide: Text.ElideRight; width: parent.width }
        }
    }
}
