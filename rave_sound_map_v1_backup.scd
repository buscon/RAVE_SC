/*
Sound map: a 2D map of audio clips. Move the mouse over a dot to hear its clip.

Made with latent_map.py. Every dot is one short clip with its embedding (latent vector). The dots are grouped
into clusters (k-means on the latent vectors; latent_map.py chooses the number, which is unrelated to the
number of latent dimensions of the model). No RAVE model and no nn.ar are needed here: this plays the saved clips.

- Hover: plays the clip under the mouse (switch off with "play on hover"). Click or drag also plays.
- Clusters: switch clusters on or off in the right panel. Only the "on" clusters can be played, and the rest
  are greyed out. "exclusive select" makes a click on a cluster switch all the others off.
- Views: islands (clusters shown as separate groups, a layout made for display), t-SNE, PCA.
  In t-SNE and PCA the clusters overlap, because the latent space is continuous.
- The text under the map gives the source of the clip, its cluster, brightness (spectral centroid) and level.

I could not run SuperCollider, so this is untested. If something fails, send me the post window text.
*/

// 1. Load the map and the clips (run once, wait for "sound map ready")
(
s.options.sampleRate = 48000;
s.options.numInputBusChannels = 2;
s.options.numOutputBusChannels = 2;
s.waitForBoot {
	~mapDir  = "/Users/marcellolussana/Documents/Software/RAVE_SC/latent_maps/";
	~mapName = "birds_dawnchorus_b2048_r48000_z8_decoder";   // use "..._audio" for a map made from recordings

	// CSV columns: x,y islands | x,y t-SNE | x,y PCA | cluster | centroid Hz | level dB | latent vector
	~rows = CSVFileReader.read(~mapDir ++ ~mapName ++ "_map.csv", true, true)
		.collect { |r| r.collect { |x| x.asFloat } };
	~grp  = ~rows.collect { |r| r[6].asInteger };
	~cen  = ~rows.collect { |r| r[7] };
	~lvl  = ~rows.collect { |r| r[8] };
	~info = File.readAllString(~mapDir ++ ~mapName ++ "_map_info.txt").split($\n);

	~gnames = Dictionary.new;
	~meta = Dictionary.new;
	File.readAllString(~mapDir ++ ~mapName ++ "_map_groups.txt").split($\n).do { |l|
		var p = l.split($|);
		if(p.size == 2) {
			if(p[0].beginsWith("#")) { ~meta[p[0]] = p[1] } { ~gnames[p[0].asInteger] = p[1] };
		};
	};
	~clipLen = ~meta["#clip_samples"].asInteger;
	~clipDur = ~clipLen / ~meta["#sr"].asFloat;
	~buf = Buffer.read(s, ~mapDir ++ ~meta["#audio"]);
	s.sync;

	SynthDef(\clip, { |buf, start = 0, dur = 0.3, amp = 0.5, gate = 1, out = 0|
		var live = Line.kr(1, 0, dur) > 0;
		var env = EnvGen.kr(Env.asr(0.004, 1, 0.04), gate * live, doneAction: 2);
		var sig = PlayBuf.ar(1, buf, BufRateScale.kr(buf), startPos: start, loop: 0);
		Out.ar(out, Pan2.ar(sig * env, 0, amp));
	}).add;
	s.sync;

	~byGroup = Dictionary.new;
	~grp.do { |g, i| ~byGroup[g] = (~byGroup[g] ? []).add(i) };
	~groupIds = ~byGroup.keys.asArray.sort;
	~active = Array.fill(~groupIds.size, true);
	~updateMask = { ~mask = ~grp.collect { |g| if(~active[g]) { 0 } { 1e6 } } };
	~updateMask.value;

	// same colours as the Python figure: 10 colours for up to 10 clusters, 20 colours for more
	~pal = (if(~groupIds.size <= 10) {
		["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
			"#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
	} {
		["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896",
			"#9467bd", "#c5b0d5", "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7",
			"#bcbd22", "#dbdb8d", "#17becf", "#9edae5"]
	}).collect { |h| Color.fromHexString(h) };
	~colorOf = { |g| ~pal[g % ~pal.size] };

	// view 0 = islands, 1 = t-SNE, 2 = PCA. Positions are scaled to 0..1 (islands keep their aspect ratio).
	~setView = { |i|
		var c = i * 2;
		var xs = ~rows.collect { |r| r[c] };
		var ys = ~rows.collect { |r| r[c + 1] };
		var xr = xs.maxItem - xs.minItem;
		var yr = ys.maxItem - ys.minItem;
		var xmid = (xs.maxItem + xs.minItem) / 2;
		var ymid = (ys.maxItem + ys.minItem) / 2;
		var xsc = if(i == 0) { max(xr, yr) } { xr };
		var ysc = if(i == 0) { max(xr, yr) } { yr };
		~xs = (xs - xmid) / xsc * 0.94 + 0.5;
		~ys = (ys - ymid) / ysc * 0.94 + 0.5;
		~centers = ~groupIds.collect { |g|
			var ids = ~byGroup[g];
			[g, ids.collect { |k| ~xs[k] }.mean, ids.collect { |k| ~ys[k] }.mean]
		};
		~cur = nil;
		if(~uv.notNil) { ~uv.refresh };
	};
	~setView.(0);

	~amp = 0.5;
	~hover = true;
	~exclusive = false;
	~lastIdx = nil;
	~lastT = 0;
	~last = nil;

	~play = { |i|
		if(~last.notNil and: { ~last.isPlaying }) { ~last.set(\gate, 0) };
		~last = Synth(\clip, [\buf, ~buf, \start, i * ~clipLen, \dur, ~clipDur, \amp, ~amp]).register;
	};

	// x, y: pixel position in the map view. trigger: play the clip. force: play even if it is the same dot.
	~at = { |x, y, trigger, force|
		var w = ~uv.bounds.width, h = ~uv.bounds.height, sq = min(w, h);
		var mx = (x - ((w - sq) / 2)) / sq;
		var my = 1 - ((y - ((h - sq) / 2)) / sq);
		var d2 = (~xs - mx).squared + (~ys - my).squared + ~mask;
		var i = d2.minIndex;
		var now = Main.elapsedTime;
		if(d2[i] < 0.0049) {   // the mouse is within 0.07 of a dot
			~cur = i;
			~label.string = "%   |   cluster %   |   brightness % Hz, level % dB".format(
				~info[i], ~grp[i], ~cen[i].round(1), ~lvl[i].round(1));
			if(trigger and: { force or: { (i != ~lastIdx) and: { (now - ~lastT) > 0.04 } } }) {
				~play.(i);
				~lastIdx = i;
				~lastT = now;
			};
		} {
			~cur = nil;
			~lastIdx = nil;   // leaving a dot: it can be played again when the mouse returns
		};
		~uv.refresh;
	};

	"sound map ready: % clips, % clusters".format(~rows.size, ~groupIds.size).postln;
};
)

// 2. Window
(
if(~win.notNil and: { ~win.isClosed.not }) { ~win.close };

~uv = UserView().minSize_(Size(560, 560)).background_(Color.white).acceptsMouseOver_(true);
~uv.drawFunc = { |v|
	var w = v.bounds.width, h = v.bounds.height, sq = min(w, h);
	var ox = (w - sq) / 2, oy = (h - sq) / 2;
	Pen.fillColor = Color.gray(0.97);
	Pen.fillRect(Rect(ox, oy, sq, sq));
	~byGroup.keysValuesDo { |g, ids|   // clusters that are off, in grey
		if(~active[g].not) {
			Pen.fillColor = Color.gray(0.8, 0.5);
			ids.do { |i| Pen.addOval(Rect(ox + (~xs[i] * sq) - 2, oy + ((1 - ~ys[i]) * sq) - 2, 4, 4)) };
			Pen.fill;
		};
	};
	~byGroup.keysValuesDo { |g, ids|   // clusters that are on, in colour
		var c = ~colorOf.(g);
		if(~active[g]) {
			Pen.fillColor = Color(c.red, c.green, c.blue, 0.85);
			ids.do { |i| Pen.addOval(Rect(ox + (~xs[i] * sq) - 2.5, oy + ((1 - ~ys[i]) * sq) - 2.5, 5, 5)) };
			Pen.fill;
		};
	};
	~centers.do { |t|
		if(~active[t[0]]) {
			Pen.stringAtPoint(t[0].asString,
				Point(ox + (t[1] * sq) - 5, oy + ((1 - t[2]) * sq) - 11),
				Font("Helvetica", 18, true), Color.black);
		};
	};
	if(~cur.notNil) {
		Pen.strokeColor = Color.black;
		Pen.width = 2;
		Pen.strokeOval(Rect(ox + (~xs[~cur] * sq) - 8, oy + ((1 - ~ys[~cur]) * sq) - 8, 16, 16));
	};
};
~uv.mouseOverAction = { |v, x, y| ~at.(x, y, ~hover, false) };
~uv.mouseMoveAction = { |v, x, y| ~at.(x, y, true, false) };   // dragging plays too
~uv.mouseDownAction = { |v, x, y| ~at.(x, y, true, true) };    // a click always plays

~label = StaticText().string_("move the mouse over the dots");

~viewPop = PopUpMenu().items_(["islands (grouped)", "t-SNE", "PCA"]).value_(0)
	.action_({ |m| ~setView.(m.value) });
~hoverBtn = Button().states_([["play on hover: off"], ["play on hover: on"]]).value_(1)
	.action_({ |b| ~hover = (b.value == 1) });

// one button per cluster: on = coloured, off = grey
~clusterBtns = ~groupIds.collect { |g|
	Button().states_([
		[~gnames[g], Color.gray(0.4), Color.gray(0.9)],
		[~gnames[g], Color.white, ~colorOf.(g)]
	]).value_(1).action_({ |b|
		if(~exclusive) {
			~active = Array.fill(~groupIds.size, false);
			~active[g] = true;
			~clusterBtns.do { |o, k| o.value = if(k == g) { 1 } { 0 } };
		} {
			~active[g] = (b.value == 1);
		};
		~updateMask.value;
		~uv.refresh;
	})
};
~exclBtn = Button().states_([["exclusive select: off"], ["exclusive select: on"]])
	.action_({ |b| ~exclusive = (b.value == 1) });
~allBtn = Button().states_([["all on"]]).action_({
	~active = Array.fill(~groupIds.size, true);
	~clusterBtns.do { |o| o.value = 1 };
	~updateMask.value;
	~uv.refresh;
});
~noneBtn = Button().states_([["all off"]]).action_({
	~active = Array.fill(~groupIds.size, false);
	~clusterBtns.do { |o| o.value = 0 };
	~updateMask.value;
	~uv.refresh;
});

~ampSpec = ControlSpec(0.1, 4, \exp);
~ampSl = Slider().orientation_(\horizontal).value_(~ampSpec.unmap(0.5))
	.action_({ |sl| ~amp = ~ampSpec.map(sl.value) });

~win = Window("RAVE sound map", Rect(60, 60, 1000, 680));
~win.layout = HLayout(
	[VLayout(~uv, ~label), stretch: 3],
	[VLayout(
		StaticText().string_("view"), ~viewPop, ~hoverBtn,
		StaticText().string_("clusters (click to switch on / off)"),
		~exclBtn, HLayout(~allBtn, ~noneBtn),
		*(~clusterBtns ++ [StaticText().string_("volume"), ~ampSl])
	), stretch: 1]
);
~win.front;
)

// Stop all sound and free the clips
// ~last.free; ~buf.free;
