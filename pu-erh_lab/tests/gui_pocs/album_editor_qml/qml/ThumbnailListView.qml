import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ListView {
    id: root
    model: albumBackend.thumbnails
    clip: true
    spacing: 8

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

        width: ListView.view.width
        height: 84
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

        RowLayout {
            anchors.fill: parent
            anchors.margins: 8
            CheckBox {
                Layout.alignment: Qt.AlignVCenter
                checked: root.isImageSelected(elementId)
                onClicked: root.imageSelectionChanged(elementId, imageId, fileName, checked)
            }
            Item {
                Layout.preferredWidth: 96
                Layout.fillHeight: true
                clip: true
                Rectangle {
                    anchors.fill: parent
                    radius: 8
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
            }
            ColumnLayout {
                Layout.fillWidth: true
                Label { Layout.fillWidth: true; text: fileName; color: "#E3DFDB"; font.pixelSize: 13; elide: Text.ElideRight }
                Label { Layout.fillWidth: true; text: cameraModel + " | " + extension + " | ISO " + iso + " | f/" + aperture + " | " + focalLength + "mm"; color: "#7B7D7C"; font.pixelSize: 11; elide: Text.ElideRight }
                Label { Layout.fillWidth: true; text: captureDate + " | tags: " + tags; color: "#7B7D7C"; font.pixelSize: 10; elide: Text.ElideRight }
            }
            ColumnLayout {
                spacing: 4
                Label { text: rating + "/5"; color: "#E3DFDB"; font.pixelSize: 12; font.weight: 700; horizontalAlignment: Text.AlignHCenter }
                Button { text: "Edit"; onClicked: albumBackend.openEditor(elementId, imageId) }
            }
        }
    }
}
