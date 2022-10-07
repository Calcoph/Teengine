use std::ops::Range;

#[derive(Debug)]
pub(crate) struct RangeTree { // TODO: Make this more efficient
    first_node: Option<RangeNode>
}

impl RangeTree {
    pub(crate) fn new() -> Self {
        Self { first_node: None }
    }

    pub(crate) fn add_num(&mut self, num: u32) {
        match &mut self.first_node {
            Some(node) => node.add_num(num),
            None => self.first_node = Some(RangeNode::new(num)),
        }
    }

    pub(crate) fn clean(&mut self) {
        self.first_node = None
    }

    pub(crate) fn contains(&self, num: &u32) -> bool {
        match &self.first_node {
            Some(node) => node.contains(num),
            None => false,
        }
    }

    pub(crate) fn get_vec(&self) -> Vec<Range<u32>> {
        match &self.first_node {
            Some(node) => node.get_vec(),
            None => Vec::new(),
        }
    }
}

#[derive(Debug)]
struct RangeNode {
    contents: Range<u32>,
    left: Option<Box<RangeNode>>,
    right: Option<Box<RangeNode>>
}

impl RangeNode {
    fn new(num: u32) -> Self {
        RangeNode {
            contents: num..num+1,
            left: None,
            right: None
        }
    }

    fn add_num(&mut self, num: u32) {
        if self.contents.end == num {
            match &self.right {
                Some(node) => {
                    if node.contents.start == num-1 {
                        let mut right = self.right.take().unwrap();
                        let new_right = right.right.take();
                        self.right = new_right;
                        self.contents.end = right.contents.end;
                    } else {
                        self.contents.end = num+1
                    }
                },
                None => self.contents.end = num+1,
            }
        } else if self.contents.start-1 == num {
            match &self.left {
                Some(node) => {
                    if node.contents.end == num {
                        let mut left = self.left.take().unwrap();
                        let new_left = left.left.take();
                        self.left = new_left;
                        self.contents.start = left.contents.start;
                    } else {
                        self.contents.start = num
                    }
                },
                None => self.contents.start = num,
            }
        } else if num > self.contents.end {
            match &mut self.right {
                Some(node) => node.add_num(num),
                None => self.right = Some(Box::new(RangeNode::new(num))),
            }
        } else { // num < self.contents.start-1
            match &mut self.left {
                Some(node) => node.add_num(num),
                None => self.left = Some(Box::new(RangeNode::new(num))),
            }
        }
    }

    fn contains(&self, num: &u32) -> bool {
        self.contents.contains(num)
        || match &self.left {
            Some(node) => node.contains(num),
            None => false,
        }
        || match &self.right {
            Some(node) => node.contains(num),
            None => false,
        }
    }

    fn get_vec(&self) -> Vec<Range<u32>> {
        let mut v = match &self.left {
            Some(node) => {
                let mut v = node.get_vec();
                v.push(self.contents.clone()); // TODO: Probably don't need to preserve self.contents, as it will be overwritten next frame. Possible optimization here
                v
            },
            None => vec![self.contents.clone()], // TODO: Probably don't need to preserve self.contents, as it will be overwritten next frame. Possible optimization here
        };

        if let Some(node) = &self.right {
            node.expand_vec(&mut v)
        };

        v
    }

    fn expand_vec(&self, v: &mut Vec<Range<u32>>) {
        if let Some(node) = &self.left {
            node.expand_vec(v)
        };

        v.push(self.contents.clone());// TODO: Probably don't need to preserve self.contents, as it will be overwritten next frame. Possible optimization here

        if let Some(node) = &self.right {
            node.expand_vec(v)
        };
    }
}
