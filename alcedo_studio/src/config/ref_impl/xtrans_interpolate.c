/* Taken from https://www.cybercom.net/~dcoffin/dcraw/dcraw.c */
/* Which itself attributes this algorithm to "Frank Markesteijn" */

#define TS 512		/* Tile Size */
#define fcol(row,col) xtrans[(row+6) % 6][(col+6) % 6]

void CLASS xtrans_interpolate (int passes)
{
  int c, d, f, g, h, i, v, ng, row, col, top, left, mrow, mcol;
  int val, ndir, pass, hm[8], avg[4], color[3][8];
  static const short orth[12] = { 1,0,0,1,-1,0,0,-1,1,0,0,1 },
	patt[2][16] = { { 0,1,0,-1,2,0,-1,0,1,1,1,-1,0,0,0,0 },
			{ 0,1,0,-2,1,0,-2,0,1,1,-2,-2,1,-1,-1,1 } },
	dir[4] = { 1,TS,TS+1,TS-1 };
  short allhex[3][3][2][8], *hex;
  ushort min, max, sgrow, sgcol;
  ushort (*rgb)[TS][TS][3], (*rix)[3], (*pix)[4];
   short (*lab)    [TS][3], (*lix)[3];
   float (*drv)[TS][TS], diff[6], tr;
   char (*homo)[TS][TS], *buffer;

  if (verbose)
    fprintf (stderr,_("%d-pass X-Trans interpolation...\n"), passes);

  cielab (0,0);
  ndir = 4 << (passes > 1);
  buffer = (char *) malloc (TS*TS*(ndir*11+6));
  merror (buffer, "xtrans_interpolate()");
  rgb  = (ushort(*)[TS][TS][3]) buffer;
  lab  = (short (*)    [TS][3])(buffer + TS*TS*(ndir*6));
  drv  = (float (*)[TS][TS])   (buffer + TS*TS*(ndir*6+6));
  homo = (char  (*)[TS][TS])   (buffer + TS*TS*(ndir*10+6));

/* Map a green hexagon around each non-green pixel and vice versa:	*/
  for (row=0; row < 3; row++)
    for (col=0; col < 3; col++)
      for (ng=d=0; d < 10; d+=2) {
	g = fcol(row,col) == 1;
	if (fcol(row+orth[d],col+orth[d+2]) == 1) ng=0; else ng++;
	if (ng == 4) { sgrow = row; sgcol = col; }
	if (ng == g+1) FORC(8) {
	  v = orth[d  ]*patt[g][c*2] + orth[d+1]*patt[g][c*2+1];
	  h = orth[d+2]*patt[g][c*2] + orth[d+3]*patt[g][c*2+1];
	  allhex[row][col][0][c^(g*2 & d)] = h + v*width;
	  allhex[row][col][1][c^(g*2 & d)] = h + v*TS;
	}
      }

/* Set green1 and green3 to the minimum and maximum allowed values:	*/
  for (row=2; row < height-2; row++)
    for (min=~(max=0), col=2; col < width-2; col++) {
      if (fcol(row,col) == 1 && (min=~(max=0))) continue;
      pix = image + row*width + col;
      hex = allhex[row % 3][col % 3][0];
      if (!max) FORC(6) {
	val = pix[hex[c]][1];
	if (min > val) min = val;
	if (max < val) max = val;
      }
      pix[0][1] = min;
      pix[0][3] = max;
      switch ((row-sgrow) % 3) {
	case 1: if (row < height-3) { row++; col--; } break;
	case 2: if ((min=~(max=0)) && (col+=2) < width-3 && row > 2) row--;
      }
    }

  for (top=3; top < height-19; top += TS-16)
    for (left=3; left < width-19; left += TS-16) {
      mrow = MIN (top+TS, height-3);
      mcol = MIN (left+TS, width-3);
      for (row=top; row < mrow; row++)
	for (col=left; col < mcol; col++)
	  memcpy (rgb[0][row-top][col-left], image[row*width+col], 6);
      FORC3 memcpy (rgb[c+1], rgb[0], sizeof *rgb);

/* Interpolate green horizontally, vertically, and along both diagonals: */
      for (row=top; row < mrow; row++)
	for (col=left; col < mcol; col++) {
	  if ((f = fcol(row,col)) == 1) continue;
	  pix = image + row*width + col;
	  hex = allhex[row % 3][col % 3][0];
	  color[1][0] = 174 * (pix[  hex[1]][1] + pix[  hex[0]][1]) -
			 46 * (pix[2*hex[1]][1] + pix[2*hex[0]][1]);
	  color[1][1] = 223 *  pix[  hex[3]][1] + pix[  hex[2]][1] * 33 +
			 92 * (pix[      0 ][f] - pix[ -hex[2]][f]);
	  FORC(2) color[1][2+c] =
		164 * pix[hex[4+c]][1] + 92 * pix[-2*hex[4+c]][1] + 33 *
		(2*pix[0][f] - pix[3*hex[4+c]][f] - pix[-3*hex[4+c]][f]);
	  FORC4 rgb[c^!((row-sgrow) % 3)][row-top][col-left][1] =
		LIM(color[1][c] >> 8,pix[0][1],pix[0][3]);
	}

      for (pass=0; pass < passes; pass++) {
	if (pass == 1)
	  memcpy (rgb+=4, buffer, 4*sizeof *rgb);

/* Recalculate green from interpolated values of closer pixels:	*/
	if (pass) {
	  for (row=top+2; row < mrow-2; row++)
	    for (col=left+2; col < mcol-2; col++) {
	      if ((f = fcol(row,col)) == 1) continue;
	      pix = image + row*width + col;
	      hex = allhex[row % 3][col % 3][1];
	      for (d=3; d < 6; d++) {
		rix = &rgb[(d-2)^!((row-sgrow) % 3)][row-top][col-left];
		val = rix[-2*hex[d]][1] + 2*rix[hex[d]][1]
		    - rix[-2*hex[d]][f] - 2*rix[hex[d]][f] + 3*rix[0][f];
		rix[0][1] = LIM(val/3,pix[0][1],pix[0][3]);
	      }
	    }
	}

/* Interpolate red and blue values for solitary green pixels:	*/
	for (row=(top-sgrow+4)/3*3+sgrow; row < mrow-2; row+=3)
	  for (col=(left-sgcol+4)/3*3+sgcol; col < mcol-2; col+=3) {
	    rix = &rgb[0][row-top][col-left];
	    h = fcol(row,col+1);
	    memset (diff, 0, sizeof diff);
	    for (i=1, d=0; d < 6; d++, i^=TS^1, h^=2) {
	      for (c=0; c < 2; c++, h^=2) {
		g = 2*rix[0][1] - rix[i<<c][1] - rix[-i<<c][1];
		color[h][d] = g + rix[i<<c][h] + rix[-i<<c][h];
		if (d > 1)
		  diff[d] += SQR (rix[i<<c][1] - rix[-i<<c][1]
				- rix[i<<c][h] + rix[-i<<c][h]) + SQR(g);
	      }
	      if (d > 1 && (d & 1))
		if (diff[d-1] < diff[d])
		  FORC(2) color[c*2][d] = color[c*2][d-1];
	      if (d < 2 || (d & 1)) {
		FORC(2) rix[0][c*2] = CLIP(color[c*2][d]/2);
		rix += TS*TS;
	      }
	    }
	  }

/* Interpolate red for blue pixels and vice versa:		*/
	for (row=top+3; row < mrow-3; row++)
	  for (col=left+3; col < mcol-3; col++) {
	    if ((f = 2-fcol(row,col)) == 1) continue;
	    rix = &rgb[0][row-top][col-left];
	    c = (row-sgrow) % 3 ? TS:1;
	    h = 3 * (c ^ TS ^ 1);
	    for (d=0; d < 4; d++, rix += TS*TS) {
	      i = d > 1 || ((d ^ c) & 1) ||
		 ((ABS(rix[0][1]-rix[c][1])+ABS(rix[0][1]-rix[-c][1])) <
		2*(ABS(rix[0][1]-rix[h][1])+ABS(rix[0][1]-rix[-h][1]))) ? c:h;
	      rix[0][f] = CLIP((rix[i][f] + rix[-i][f] +
		  2*rix[0][1] - rix[i][1] - rix[-i][1])/2);
	    }
	  }

/* Fill in red and blue for 2x2 blocks of green:		*/
	for (row=top+2; row < mrow-2; row++) if ((row-sgrow) % 3)
	  for (col=left+2; col < mcol-2; col++) if ((col-sgcol) % 3) {
	    rix = &rgb[0][row-top][col-left];
	    hex = allhex[row % 3][col % 3][1];
	    for (d=0; d < ndir; d+=2, rix += TS*TS)
	      if (hex[d] + hex[d+1]) {
		g = 3*rix[0][1] - 2*rix[hex[d]][1] - rix[hex[d+1]][1];
		for (c=0; c < 4; c+=2) rix[0][c] =
			CLIP((g + 2*rix[hex[d]][c] + rix[hex[d+1]][c])/3);
	      } else {
		g = 2*rix[0][1] - rix[hex[d]][1] - rix[hex[d+1]][1];
		for (c=0; c < 4; c+=2) rix[0][c] =
			CLIP((g + rix[hex[d]][c] + rix[hex[d+1]][c])/2);
	      }
	  }
      }
      rgb = (ushort(*)[TS][TS][3]) buffer;
      mrow -= top;
      mcol -= left;

/* Convert to CIELab and differentiate in all directions:	*/
      for (d=0; d < ndir; d++) {
	for (row=2; row < mrow-2; row++)
	  for (col=2; col < mcol-2; col++)
	    cielab (rgb[d][row][col], lab[row][col]);
	for (f=dir[d & 3],row=3; row < mrow-3; row++)
	  for (col=3; col < mcol-3; col++) {
	    lix = &lab[row][col];
	    g = 2*lix[0][0] - lix[f][0] - lix[-f][0];
	    drv[d][row][col] = SQR(g)
	      + SQR((2*lix[0][1] - lix[f][1] - lix[-f][1] + g*500/232))
	      + SQR((2*lix[0][2] - lix[f][2] - lix[-f][2] - g*500/580));
	  }
      }

/* Build homogeneity maps from the derivatives:			*/
      memset(homo, 0, ndir*TS*TS);
      for (row=4; row < mrow-4; row++)
	for (col=4; col < mcol-4; col++) {
	  for (tr=FLT_MAX, d=0; d < ndir; d++)
	    if (tr > drv[d][row][col])
		tr = drv[d][row][col];
	  tr *= 8;
	  for (d=0; d < ndir; d++)
	    for (v=-1; v <= 1; v++)
	      for (h=-1; h <= 1; h++)
		if (drv[d][row+v][col+h] <= tr)
		  homo[d][row][col]++;
	}

/* Average the most homogenous pixels for the final result:	*/
      if (height-top < TS+4) mrow = height-top+2;
      if (width-left < TS+4) mcol = width-left+2;
      for (row = MIN(top,8); row < mrow-8; row++)
	for (col = MIN(left,8); col < mcol-8; col++) {
	  for (d=0; d < ndir; d++)
	    for (hm[d]=0, v=-2; v <= 2; v++)
	      for (h=-2; h <= 2; h++)
		hm[d] += homo[d][row+v][col+h];
	  for (d=0; d < ndir-4; d++)
	    if (hm[d] < hm[d+4]) hm[d  ] = 0; else
	    if (hm[d] > hm[d+4]) hm[d+4] = 0;
	  for (max=hm[0],d=1; d < ndir; d++)
	    if (max < hm[d]) max = hm[d];
	  max -= max >> 3;
	  memset (avg, 0, sizeof avg);
	  for (d=0; d < ndir; d++)
	    if (hm[d] >= max) {
	      FORC3 avg[c] += rgb[d][row][col][c];
	      avg[3]++;
	    }
	  FORC3 image[(row+top)*width+col+left][c] = avg[c]/avg[3];
	}
    }
  free(buffer);
  border_interpolate(8);
}
#undef fcol
