#include "videofile.h"
int main(int argc, char *agrv[]) {
  VideoFile video_file(agrv[1]);
  video_file.Display(125, 640);
  return 0;
}
